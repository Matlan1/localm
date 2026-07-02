# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.discover - HF search, quant parsing, VRAM fit badges.
All HuggingFace calls are mocked; no real network."""

import pytest

from localm.discover import (
    DiscoverError, _quant_of, fit_label, hf_gguf_files, hf_search, vram_info,
)


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    """Deterministic policy: discovery only checks the off switch."""
    monkeypatch.setenv("LOCALM_NET_MODE", "ask")


def _mock_fetch(monkeypatch, payload, seen=None):
    """Patch netpolicy.safe_fetch_bytes (discover now routes through it for SSRF
    pinning + redirect re-validation) to return *payload* as JSON bytes. When
    *seen* is given, capture the request's query params (now encoded into the URL)
    so the param-asserting tests keep working."""
    import json as _json
    import urllib.parse

    def fake(url, **kw):
        if seen is not None:
            seen.update(dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query)))
        return url, "application/json", _json.dumps(payload).encode("utf-8")
    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)


# ------------------------------------------------------------------ #
#  Quant label parsing                                                 #
# ------------------------------------------------------------------ #

class TestQuantParsing:
    @pytest.mark.parametrize("name,quant", [
        ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
        ("model-Q8_0.gguf", "Q8_0"),
        ("model-IQ4_XS.gguf", "IQ4_XS"),
        ("model-Q6_K.gguf", "Q6_K"),
        ("model-Q2_K_L.gguf", "Q2_K_L"),
        ("model-f16.gguf", "F16"),
        ("model-BF16.gguf", "BF16"),
        ("mystery-model.gguf", ""),
    ])
    def test_quant_of(self, name, quant):
        assert _quant_of(name) == quant


# ------------------------------------------------------------------ #
#  Search                                                              #
# ------------------------------------------------------------------ #

class TestSearch:
    def test_search_parses_and_sends_gguf_filter(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "org/model-GGUF", "downloads": 5, "likes": 2,
             "lastModified": "2026-01-01"},
            {"modelId": "org/other"},
        ], seen)
        results = hf_search("qwen 7b", limit=5)
        assert seen["filter"] == "gguf"
        assert seen["search"] == "qwen 7b"
        assert seen["sort"] == "downloads"
        assert results[0] == {"id": "org/model-GGUF", "downloads": 5,
                              "likes": 2, "updated": "2026-01-01"}
        assert results[1]["id"] == "org/other"   # modelId fallback

    def test_empty_query_is_popular_list(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [], seen)
        hf_search("", limit=3)
        assert "search" not in seen        # no query → most-downloaded view
        assert seen["limit"] == "3"

    def test_net_off_blocks(self, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        with pytest.raises(DiscoverError, match="net_mode=off"):
            hf_search("anything")

    def test_network_failure_wrapped(self, monkeypatch):
        def boom(url, **kw):
            raise ConnectionError("refused")
        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", boom)
        with pytest.raises(DiscoverError, match="request failed"):
            hf_search("x")

    def test_ssrf_policy_refusal_wrapped(self, monkeypatch):
        """A netpolicy refusal (e.g. a rebind to a private address) surfaces as a
        DiscoverError, proving discovery now goes through the SSRF-checked path."""
        from localm.netpolicy import NetworkPolicyError

        def refuse(url, **kw):
            raise NetworkPolicyError("resolves to a private address")
        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", refuse)
        with pytest.raises(DiscoverError, match="request failed"):
            hf_search("x")


# ------------------------------------------------------------------ #
#  File listing                                                        #
# ------------------------------------------------------------------ #

_TREE = [
    {"path": "README.md", "size": 100},
    {"path": "model-Q4_K_M.gguf", "size": 4_000},
    {"path": "model-Q8_0.gguf", "lfs": {"size": 8_000}},
    {"path": "big-F16-00001-of-00003.gguf", "size": 10_000},
    {"path": "big-F16-00002-of-00003.gguf", "size": 10_000},
    {"path": "big-F16-00003-of-00003.gguf", "size": 5_000},
]


class TestFiles:
    def test_parse_group_and_sort(self, monkeypatch):
        _mock_fetch(monkeypatch, _TREE)
        files = hf_gguf_files("org/repo")
        assert [f["file"] for f in files] == [
            "model-Q4_K_M.gguf",            # 4k
            "model-Q8_0.gguf",              # 8k (lfs size used)
            "big-F16-00001-of-00003.gguf",  # 25k summed, first part is the spec
        ]
        split = files[-1]
        assert split["size_bytes"] == 25_000
        assert split["n_parts"] == 3
        assert split["quant"] == "F16"
        assert files[1]["size_bytes"] == 8_000

    def test_no_gguf_files_hints_full_pull(self, monkeypatch):
        _mock_fetch(monkeypatch, [{"path": "config.json"}])
        with pytest.raises(DiscoverError, match="pull it whole"):
            hf_gguf_files("org/transformers-repo")

    @pytest.mark.parametrize("bad", ["", "no-slash", "a/b/c", "a b/c"])
    def test_invalid_repo_id_rejected(self, monkeypatch, bad):
        _mock_fetch(monkeypatch, [])
        with pytest.raises(DiscoverError, match="repo id"):
            hf_gguf_files(bad)


# ------------------------------------------------------------------ #
#  VRAM fit                                                            #
# ------------------------------------------------------------------ #

class TestFit:
    def test_unknown_vram_no_badge(self):
        assert fit_label(5_000_000_000, None) == ""
        assert fit_label(0, 16_000_000_000) == ""

    def test_thresholds(self):
        total = 16_000_000_000
        # need = size*1.10 + 1.5e9; 0.85*total = 13.6e9 → size ≈ 11e9 fits
        assert fit_label(int(10e9), total) == "fits"
        assert fit_label(int(12.5e9), total) == "tight"   # need ≈ 15.25e9 ≤ 16e9
        assert fit_label(int(14e9), total) == "too-big"   # need ≈ 16.9e9

    def test_vram_info_shape(self):
        info = vram_info()
        assert isinstance(info, dict)
        if info:
            assert info.get("total", 0) > 0
