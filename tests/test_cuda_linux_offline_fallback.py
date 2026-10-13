# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Linux CUDA build resolves from its pinned URL and checksum when the
release listing cannot be read (offline, API rate limit), and never otherwise."""

import click
import pytest

import localm.setup_llama as sl
from localm.setup_llama.assets import _resolve_linux_cuda_asset
from localm.setup_llama.pins import _CUDA_LINUX_PINNED_SHA256, _CUDA_LINUX_REPO, _PINNED_TAG


def _listing(monkeypatch, assets):
    monkeypatch.setattr(sl, "_release_assets", lambda tag, repo=None: assets)


@pytest.mark.parametrize("line, name", [
    ("cuda-12", f"llama-{_PINNED_TAG}-bin-ubuntu-cuda-x64.tar.gz"),
    ("cuda-13", f"llama-{_PINNED_TAG}-bin-ubuntu-cuda-13-x64.tar.gz"),
])
def test_an_unreadable_listing_resolves_the_pinned_build(monkeypatch, capsys, line, name):
    _listing(monkeypatch, [])
    url, sha, tag = _resolve_linux_cuda_asset("cuda", line, _PINNED_TAG)
    assert url == f"https://github.com/{_CUDA_LINUX_REPO}/releases/download/{_PINNED_TAG}/{name}"
    assert sha == _CUDA_LINUX_PINNED_SHA256[name]
    assert tag == _PINNED_TAG
    assert "pinned" in capsys.readouterr().out


def test_an_unreadable_listing_for_an_unpinned_tag_still_raises(monkeypatch):
    _listing(monkeypatch, [])
    with pytest.raises(click.ClickException, match="no Linux CUDA build found"):
        _resolve_linux_cuda_asset("cuda", "cuda-12", "b1")


def test_a_readable_listing_without_the_asset_does_not_fall_back(monkeypatch):
    _listing(monkeypatch, [{"name": "llama-" + _PINNED_TAG + "-bin-ubuntu-vulkan-x64.tar.gz",
                            "browser_download_url": "https://example.invalid/x"}])
    with pytest.raises(click.ClickException, match="no Linux CUDA build found"):
        _resolve_linux_cuda_asset("cuda", "cuda-12", _PINNED_TAG)


def test_a_readable_listing_wins_over_the_pinned_table(monkeypatch):
    name = f"llama-{_PINNED_TAG}-bin-ubuntu-cuda-x64.tar.gz"
    _listing(monkeypatch, [{"name": name, "browser_download_url": "https://example.invalid/live",
                            "digest": "sha256:" + "ab" * 32}])
    url, sha, _tag = _resolve_linux_cuda_asset("cuda", "cuda-12", _PINNED_TAG)
    assert url == "https://example.invalid/live" and sha == "ab" * 32


def test_the_pinned_table_names_the_pinned_tag_and_both_lines():
    assert set(_CUDA_LINUX_PINNED_SHA256) == {
        f"llama-{_PINNED_TAG}-bin-ubuntu-cuda-x64.tar.gz",
        f"llama-{_PINNED_TAG}-bin-ubuntu-cuda-13-x64.tar.gz"}
    assert all(len(v) == 64 and set(v) <= set("0123456789abcdef")
               for v in _CUDA_LINUX_PINNED_SHA256.values())
