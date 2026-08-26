# SPDX-License-Identifier: AGPL-3.0-or-later
"""A link-local / cloud-metadata comfy_api_url is refused so an ADMIN-set
api_url cannot turn the comfy control calls into an SSRF probe of cloud
metadata. Loopback / LAN / public are allowed - a real ComfyUI runs on any."""

from localm.media import comfy_client as c

_LOOPBACK = "http://127.0.0.1:8188"


def test_sanitize_rejects_link_local_metadata():
    assert c.sanitize_comfy_url("http://169.254.169.254:8188") == _LOOPBACK   # cloud metadata
    assert c.sanitize_comfy_url("http://169.254.1.5/") == _LOOPBACK


def test_sanitize_allows_loopback_lan_public():
    assert c.sanitize_comfy_url(_LOOPBACK) == _LOOPBACK
    assert c.sanitize_comfy_url("http://192.168.1.50:8188") == "http://192.168.1.50:8188"
    assert c.sanitize_comfy_url("http://10.0.0.9:8188") == "http://10.0.0.9:8188"


def test_default_api_url_refuses_metadata_config(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"comfy_api_url": "http://169.254.169.254:8188"})
    assert c.default_api_url() == _LOOPBACK


def test_default_api_url_keeps_lan_config(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"comfy_api_url": "http://192.168.1.50:8188"})
    assert c.default_api_url() == "http://192.168.1.50:8188"


# --------------------------------------------------------------------------- #
# settings() sanitises the RESOLVED api_url on every media plugin, so a
# per-plugin or global api_url cannot reach the outbound comfy calls by
# short-circuiting default_api_url()'s own guard.
# --------------------------------------------------------------------------- #

from localm.plugins.builtin.image import backend as _image_backend    # noqa: E402
from localm.plugins.builtin.music import backend as _music_backend    # noqa: E402
from localm.plugins.builtin.video import backend as _video_backend    # noqa: E402

_METADATA = "http://169.254.169.254:8188"


def _plugin_cfg(name: str, comfy_block: dict) -> dict:
    return {"plugins": {name: {"comfy": comfy_block}}}


def test_image_settings_sanitizes_per_plugin_api_url(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    s = _image_backend.settings(_plugin_cfg("image", {"api_url": _METADATA}))
    assert s["api_url"] == _LOOPBACK


def test_image_settings_sanitizes_global_comfy_api_url(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    # image also honours the legacy global comfy_api_url as a fallback.
    s = _image_backend.settings({"comfy_api_url": _METADATA})
    assert s["api_url"] == _LOOPBACK


def test_music_settings_sanitizes_per_plugin_api_url(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    s = _music_backend.settings(_plugin_cfg("music", {"api_url": _METADATA}))
    assert s["api_url"] == _LOOPBACK


def test_video_settings_sanitizes_per_plugin_api_url(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    s = _video_backend.settings(_plugin_cfg("video", {"api_url": _METADATA}))
    assert s["api_url"] == _LOOPBACK


def test_image_settings_keeps_lan_per_plugin_api_url(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    lan = "http://192.168.1.50:8188"
    s = _image_backend.settings(_plugin_cfg("image", {"api_url": lan}))
    assert s["api_url"] == lan


def test_sanitize_fails_closed_on_unparseable_url(caplog):
    # urlparse raises "Invalid IPv6 URL" on an unclosed bracket. The guard
    # refuses (fail closed) and logs why.
    import logging
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert c.sanitize_comfy_url("http://[::1") == _LOOPBACK
    assert "could not be validated" in caplog.text
