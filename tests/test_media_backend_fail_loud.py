# SPDX-License-Identifier: AGPL-3.0-or-later
"""A configured media `backend` name that cannot be loaded must surface a warning,
never fall back to ComfyUI silently. The fallback itself stays - a typo must not
hard-crash a generate - but it says so.
"""

import importlib

import pytest

from localm.plugins import media_config

IMAGE_PKG = "localm.plugins.builtin.image"
MUSIC_PKG = "localm.plugins.builtin.music"
VIDEO_PKG = "localm.plugins.builtin.video"


# --------------------------------------------------------------------------- #
#  media_config helpers                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["comfy", "", "  COMFY  ", None])
def test_no_warning_for_comfy_or_empty(name):
    assert media_config.backend_unavailable_warning(IMAGE_PKG, name) is None


@pytest.mark.parametrize("pkg", [IMAGE_PKG, MUSIC_PKG, VIDEO_PKG])
def test_warning_for_unknown_backend_names_the_backend(pkg):
    w = media_config.backend_unavailable_warning(pkg, "a1111")
    assert w is not None
    assert "a1111" in w
    assert "comfyui" in w.lower()


def test_combine_warnings():
    assert media_config.combine_warnings(None, None, "") is None
    assert media_config.combine_warnings("a", None, "b") == "a; b"
    assert media_config.combine_warnings("only") == "only"


# --------------------------------------------------------------------------- #
#  Each plugin's settings() folds the warning in                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("plugin", ["image", "music", "video"])
def test_settings_warns_on_unknown_backend(plugin):
    backend = importlib.import_module(f"localm.plugins.builtin.{plugin}.backend")
    cfg = {"plugins": {plugin: {"backend": "nope"}}}
    s = backend.settings(cfg)
    assert s["backend"] == "nope"          # the configured value is reported
    assert s["warning"] is not None
    assert "nope" in s["warning"]


@pytest.mark.parametrize("plugin", ["image", "music", "video"])
def test_settings_no_backend_warning_for_default(plugin):
    backend = importlib.import_module(f"localm.plugins.builtin.{plugin}.backend")
    s = backend.settings({})               # default backend = comfy
    assert s["backend"] == "comfy"
    # The default path carries no unknown-backend note (other config warnings,
    # like share-config, are unaffected - there are none here).
    assert s["warning"] is None


def test_settings_combines_share_config_and_backend_warnings():
    """A share-config warning and an unknown-backend warning must both survive -
    neither silently drops the other."""
    backend = importlib.import_module("localm.plugins.builtin.image.backend")
    cfg = {"plugins": {"image": {"backend": "nope", "use_config_from": "music"}}}
    s = backend.settings(cfg)
    # music is not active here -> share-config falls back with a warning,
    # AND the unknown backend warns.
    assert s["warning"] is not None
    assert "nope" in s["warning"]
    assert "music" in s["warning"]


# --------------------------------------------------------------------------- #
#  Best-effort fallback is preserved (no hard crash)                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("plugin", ["image", "music", "video"])
def test_unknown_backend_still_routes_to_comfy(plugin, monkeypatch):
    backend = importlib.import_module(f"localm.plugins.builtin.{plugin}.backend")
    assert backend._impl({"backend": "nope"}) is backend._COMFY_REF
    assert backend._impl({"backend": "comfy"}) is backend._COMFY_REF
    assert backend._impl({}) is backend._COMFY_REF

    # Forces the real failure mode, so load_backend is actually invoked instead of
    # skipped because the plugin dir lacks a 'nope.py'.
    calls = []

    def boom(package, name):
        calls.append((package, name))
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(media_config, "load_backend", boom)
    assert backend._impl({"backend": "ghost"}) is backend._COMFY_REF
    assert calls == [(f"localm.plugins.builtin.{plugin}", "ghost")]
