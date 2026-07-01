# SPDX-License-Identifier: AGPL-3.0-or-later
"""CHK-COMFY-APIURL: a link-local / cloud-metadata comfy_api_url is refused so an
ADMIN-set api_url cannot turn the comfy control calls into an SSRF probe of cloud
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
