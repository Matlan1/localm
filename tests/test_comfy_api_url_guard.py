# SPDX-License-Identifier: AGPL-3.0-or-later
"""A link-local / cloud-metadata comfy_api_url is refused so an ADMIN-set
api_url cannot turn the comfy control calls into an SSRF probe of cloud
metadata. Loopback / LAN / public are allowed - a real ComfyUI runs on any."""

import pytest

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


# --------------------------------------------------------------------------- #
# NEW-MULTIPLE-SITES-CITING-AGENTS (scoped instance): the guard's fallback was
# only ever surfaced to the debug log, invisible without --debug. settings()
# has an actual user-facing warning channel (piped to job.push in plug.py) -
# the checked variant must feed it.
# --------------------------------------------------------------------------- #

def test_sanitize_checked_reports_no_warning_on_a_clean_url():
    url, warning = c.sanitize_comfy_url_checked(_LOOPBACK)
    assert url == _LOOPBACK
    assert warning is None


def test_sanitize_checked_reports_a_warning_on_fallback():
    url, warning = c.sanitize_comfy_url_checked(_METADATA)
    assert url == _LOOPBACK
    assert warning and "link-local" in warning


def test_image_settings_surfaces_the_apiurl_guard_warning(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    s = _image_backend.settings(_plugin_cfg("image", {"api_url": _METADATA}))
    assert s["api_url"] == _LOOPBACK
    assert s["warning"] and "link-local" in s["warning"]


def test_music_settings_surfaces_the_apiurl_guard_warning(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    s = _music_backend.settings(_plugin_cfg("music", {"api_url": _METADATA}))
    assert s["api_url"] == _LOOPBACK
    assert s["warning"] and "link-local" in s["warning"]


def test_video_settings_surfaces_the_apiurl_guard_warning(monkeypatch):
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    s = _video_backend.settings(_plugin_cfg("video", {"api_url": _METADATA}))
    assert s["api_url"] == _LOOPBACK
    assert s["warning"] and "link-local" in s["warning"]


def test_image_settings_no_warning_on_lan_api_url(monkeypatch):
    # Control: a normal LAN url must not manufacture a warning.
    monkeypatch.delenv("FLUX_API_URL", raising=False)
    lan = "http://192.168.1.50:8188"
    s = _image_backend.settings(_plugin_cfg("image", {"api_url": lan}))
    assert s["api_url"] == lan
    assert not s["warning"]


# --------------------------------------------------------------------------- #
# The guard delegates its link-local classification to netpolicy, which
# normalizes numeric/short-form IPv4 literals (netpolicy._literal_ipv4)
# BEFORE classifying - a second, weaker implementation living in this module
# used to miss every one of these spellings while catching the dotted form.
# --------------------------------------------------------------------------- #

# Each of these is a different obfuscation of 169.254.169.254 (cloud metadata).
_LINK_LOCAL_NUMERIC_FORMS = [
    "0xa9fea9fe",              # dotless hex
    "2852039166",              # dotless decimal
    "0251.0376.0251.0376",     # octal, every octet
    "169.254.169.254",         # dotted (control: the un-obfuscated form)
]


@pytest.mark.parametrize("host", _LINK_LOCAL_NUMERIC_FORMS)
def test_sanitize_rejects_every_numeric_spelling_of_link_local(host):
    url = f"http://{host}:8188"
    assert c.sanitize_comfy_url(url) == _LOOPBACK
    sanitized, warning = c.sanitize_comfy_url_checked(url)
    assert sanitized == _LOOPBACK
    assert warning and "link-local" in warning


def test_sanitize_still_allows_the_numeric_loopback_literal():
    """2130706433 is a NUMERIC SPELLING of 127.0.0.1 (loopback), not
    link-local. The guard must stay narrow - refusing every numeric spelling
    of link-local must not widen into refusing loopback/private/public
    addresses too."""
    url = "http://2130706433:8188/"
    assert c.sanitize_comfy_url(url) == url


def test_sanitize_runs_the_shape_check_before_classifying():
    """A raw backslash in the authority is a parser-differential SSRF
    smuggle (netpolicy.check_url_shape's own docstring: urlparse and an HTTP
    client disagree on where the backslash terminates the authority), so it
    must be refused before this guard ever decides local vs remote - not
    silently classified using whichever host urlparse happens to extract."""
    sanitized, warning = c.sanitize_comfy_url_checked(
        "http://127.0.0.1\\@169.254.169.254/")
    assert sanitized == _LOOPBACK
    assert warning and "could not be validated" in warning


def test_host_is_link_local_helper_is_gone():
    """The second, weaker classifier must not quietly come back."""
    assert not hasattr(c, "_host_is_link_local")
