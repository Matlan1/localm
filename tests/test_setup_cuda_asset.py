# SPDX-License-Identifier: AGPL-3.0-or-later
"""The CUDA build matcher does not pick the cudart runtime zip.

The build and the CUDA runtime share the "...bin-win-cuda-12.x..." name fragment
(the runtime is cudart-llama-bin-win-cuda-12.4-x64.zip) and the runtime is often
listed FIRST by the GitHub API, so a plain substring match resolves the BUILD to
the runtime-only zip (CUDA DLLs, no llama.dll). Both build-resolution paths
(_resolve_backend_url and _resolve_cuda_pair) exclude cudart from the build
match.
"""

import io
import json

from localm import setup_llama as sl
from tests._fake_https import patch_https_transport

# Realistic upstream listing: the cudart runtime is listed BEFORE the build,
# both carry "bin-win-cuda-12".
ASSETS = [
    {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://example/cudart-cuda-12.zip", "size": 390_000_000},
    {"name": "llama-b9842-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "https://example/llama-cuda-12.zip", "size": 50_000_000},
    {"name": "llama-b9842-bin-win-vulkan-x64.zip",
     "browser_download_url": "https://example/llama-vulkan.zip", "size": 40_000_000},
]


def test_resolve_cuda_pair_build_is_llama_not_cudart(monkeypatch):
    monkeypatch.setattr(sl, "_release_assets", lambda tag: list(ASSETS))
    build, cudart = sl._resolve_cuda_pair("b9842")
    assert build is not None and cudart is not None
    assert build["name"].startswith("llama-")          # the BUILD, not the runtime
    assert "cudart" not in build["name"]
    assert build["browser_download_url"] == "https://example/llama-cuda-12.zip"
    assert cudart["name"].startswith("cudart-")         # the runtime bundle
    assert cudart["browser_download_url"] == "https://example/cudart-cuda-12.zip"


class _FakeHTTP:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self._b)

    def __exit__(self, *a):
        return False


def test_resolve_backend_url_cuda_skips_cudart(monkeypatch):
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9842")
    patch_https_transport(monkeypatch,
                        lambda req, timeout=None, context=None: _FakeHTTP({"assets": ASSETS}))
    url = sl._resolve_backend_url("cuda")
    assert url == "https://example/llama-cuda-12.zip"   # NOT the cudart zip
    assert "cudart" not in url


def test_resolve_backend_url_vulkan_unaffected(monkeypatch):
    monkeypatch.setattr(sl, "_platform_key", lambda: "win32")
    monkeypatch.setattr(sl, "_latest_tag", lambda: "b9842")
    patch_https_transport(monkeypatch,
                        lambda req, timeout=None, context=None: _FakeHTTP({"assets": ASSETS}))
    assert sl._resolve_backend_url("vulkan") == "https://example/llama-vulkan.zip"


def test_pick_asset_exclude_filters_cudart():
    # Unit: the exclude param drops the runtime even when it sorts first.
    build = sl._pick_asset(ASSETS, "bin-win-cuda-12", exclude=("cudart",))
    assert build is not None and build["name"].startswith("llama-")
    # Without exclude, the first match IS the cudart zip.
    assert sl._pick_asset(ASSETS, "bin-win-cuda-12")["name"].startswith("cudart-")
