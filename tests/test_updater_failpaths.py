# SPDX-License-Identifier: AGPL-3.0-or-later
"""Failure-path coverage for the self-updater: the backend-detection fallback and the
download failures. Everything uses injectables, so no test touches the real install or
the network."""


import pytest

from localm import updater
from localm.bugreport import LocalmError

# --------------------------- _installed_backend ---------------------------
# _installed_backend() PRESERVES whatever backend is already installed
# (setup_llama's on-disk marker) rather than re-deriving one from the current
# hardware-recommendation policy.

def test_installed_backend_preserves_what_is_actually_installed(tmp_path, monkeypatch):
    """An install with amd-rocm provisioned stays on amd-rocm across an update,
    even when the CURRENT hardware recommendation names something else.
    Exercises the REAL on-disk marker (setup_llama._record_provisioned_backend /
    _provisioned_backend) rather than a mocked return value.

    Asserts recommended_install_backend was never CALLED, from OUTSIDE the call
    with a plain counter: _installed_backend() wraps both its lookups in a broad
    except, which would swallow an exception raised as a mock side_effect."""
    from localm import hwdetect, setup_llama
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", lambda: tmp_path)
    setup_llama._record_provisioned_backend(tmp_path, "amd-rocm")

    calls = []
    monkeypatch.setattr(
        hwdetect, "recommended_install_backend",
        lambda *a, **k: calls.append(1) or "cuda-from-a-changed-recommendation")

    assert updater._installed_backend() == "amd-rocm"
    assert calls == [], ("must not re-derive a recommendation when a backend "
                         "is already installed")


def test_installed_backend_falls_back_to_recommendation_when_nothing_provisioned(
        tmp_path, monkeypatch):
    """No marker on disk (a fresh install, or one predating the marker) means
    there is nothing to preserve, so it falls back to the current hardware
    recommendation."""
    from localm import hwdetect, setup_llama
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", lambda: tmp_path)   # empty: no marker
    monkeypatch.setattr(hwdetect, "recommended_install_backend",
                        lambda *a, **k: "cuda-from-the-policy")
    assert updater._installed_backend() == "cuda-from-the-policy"


def test_installed_backend_falls_back_to_vulkan_when_everything_fails(monkeypatch):
    """Both the marker read AND the hardware detection raise -> the universal
    default, so a detection failure does not break an update."""
    from localm import hwdetect, setup_llama

    def boom_lib_dir():
        raise RuntimeError("runtime lib dir blew up")
    monkeypatch.setattr(setup_llama, "_repo_runtime_lib", boom_lib_dir)

    def boom_recommend(*a, **k):
        raise RuntimeError("detection blew up")
    monkeypatch.setattr(hwdetect, "recommended_install_backend", boom_recommend)

    assert updater._installed_backend() == "vulkan"


# --------------------- download(): real urllib error paths ----------------
# The default opener (opener=None) has its own HTTP/URL error handling.

def _config(monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"bugreport_upload_url": "https://w", "bugreport_upload_token": "t"})


# download() streams through a build_opener() (an HTTPS-only redirect handler),
# so these mock OpenerDirector.open, the real transport.
def test_download_http_error_raises_localmerror(tmp_path, monkeypatch):
    import urllib.request
    _config(monkeypatch)

    def boom(self, req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    with pytest.raises(LocalmError, match="download failed"):
        updater.download(7, tmp_path / "b.zip")


def test_download_urlerror_raises_localmerror(tmp_path, monkeypatch):
    import urllib.request
    _config(monkeypatch)

    def boom(self, req, timeout=None):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    with pytest.raises(LocalmError, match="could not download"):
        updater.download(7, tmp_path / "b.zip")


def test_download_non_2xx_status_raises(tmp_path, monkeypatch):
    import urllib.request
    _config(monkeypatch)

    class _Resp:
        status = 500
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n): return b""
    monkeypatch.setattr(urllib.request.OpenerDirector, "open",
                        lambda self, req, timeout=None: _Resp())
    with pytest.raises(LocalmError, match="download failed"):
        updater.download(7, tmp_path / "b.zip")


def test_download_unconfigured_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    with pytest.raises(LocalmError, match="not configured"):
        updater.download(7, tmp_path / "b.zip")
