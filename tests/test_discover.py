# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.discover - HF search, quant parsing, VRAM fit badges.
All HuggingFace calls are mocked; no real network."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm.discover import (
    DiscoverError, _quant_of, apply_main_gpu, fit_label, hf_backend_available,
    hf_gguf_files, hf_param_bytes, hf_search, list_gpus, resolve_main_gpu_index,
    vram_info,
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
            {"modelId": "org/other", "downloads": 1},
        ], seen)
        results = hf_search("qwen 7b", limit=5)   # default formats = gguf only
        assert seen["filter"] == "gguf"
        assert seen["search"] == "qwen 7b"
        assert seen["sort"] == "downloads"
        assert results[0] == {"id": "org/model-GGUF", "downloads": 5,
                              "likes": 2, "updated": "2026-01-01",
                              "formats": ["gguf"]}
        assert results[1]["id"] == "org/other"   # modelId fallback

    def test_empty_query_is_popular_list(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [], seen)
        hf_search("", limit=3)
        assert "search" not in seen        # no query → most-downloaded view
        assert seen["limit"] == "3"

    def test_hf_format_uses_transformers_filter(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "org/hf-model", "downloads": 9, "likes": 3,
             "lastModified": "2026-02-02"},
        ], seen)
        results = hf_search("gemma", limit=5, formats=["hf"])
        assert seen["filter"] == "transformers"   # hf -> transformers library tag
        assert results[0]["id"] == "org/hf-model"
        assert results[0]["formats"] == ["hf"]

    def test_both_formats_merge_dedupe_and_interleave(self, monkeypatch):
        """gguf and hf are queried separately; a repo present in both keeps a
        merged formats list, and the two lists are round-robin interleaved."""
        import json as _json
        import urllib.parse

        gguf_payload = [
            {"id": "org/both", "downloads": 100, "likes": 5},
            {"id": "org/only-gguf", "downloads": 40, "likes": 1},
        ]
        hf_payload = [
            {"id": "org/both", "downloads": 100, "likes": 5},   # same repo
            {"id": "org/only-hf", "downloads": 70, "likes": 2},
        ]

        def fake(url, **kw):
            filt = dict(urllib.parse.parse_qsl(
                urllib.parse.urlparse(url).query)).get("filter")
            payload = gguf_payload if filt == "gguf" else hf_payload
            return url, "application/json", _json.dumps(payload).encode("utf-8")

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        results = hf_search("x", limit=10, formats=["gguf", "hf"])
        by_id = {r["id"]: r for r in results}
        # de-duped: org/both appears once, tagged with both formats
        assert sorted(by_id["org/both"]["formats"]) == ["gguf", "hf"]
        assert by_id["org/only-gguf"]["formats"] == ["gguf"]
        assert by_id["org/only-hf"]["formats"] == ["hf"]
        # interleaved by per-format rank: gguf[0], hf[0], gguf[1]
        assert [r["id"] for r in results] == [
            "org/both", "org/only-hf", "org/only-gguf"]

    def test_interleave_keeps_gguf_visible_when_hf_dominates(self, monkeypatch):
        """The real-world failure the interleave fixes: HF repos routinely have
        far higher download counts than GGUF repacks, so a plain sort-by-downloads
        would push GGUF out of the top `limit` and a 'show GGUF' toggle could
        return zero GGUF. Interleaving guarantees both formats stay visible."""
        import json as _json
        import urllib.parse

        gguf = [{"id": "org/g1", "downloads": 5}, {"id": "org/g2", "downloads": 4}]
        hf = [{"id": "org/h1", "downloads": 1000}, {"id": "org/h2", "downloads": 999}]

        def fake(url, **kw):
            filt = dict(urllib.parse.parse_qsl(
                urllib.parse.urlparse(url).query)).get("filter")
            payload = gguf if filt == "gguf" else hf
            return url, "application/json", _json.dumps(payload).encode("utf-8")

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        results = hf_search("x", limit=3, formats=["gguf", "hf"])
        fmts = {f for r in results for f in r["formats"]}
        assert "gguf" in fmts and "hf" in fmts   # both visible despite HF's downloads
        # leads with each format's most popular, interleaved: g1, h1, g2
        assert [r["id"] for r in results] == ["org/g1", "org/h1", "org/g2"]

    def test_no_valid_format_errors(self, monkeypatch):
        _mock_fetch(monkeypatch, [])
        with pytest.raises(DiscoverError, match="model format"):
            hf_search("x", formats=["bogus"])

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
#  HF backend availability probe                                      #
# ------------------------------------------------------------------ #

class TestBackendAvailable:
    def test_true_when_both_present(self, monkeypatch):
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name: object())        # every module resolves
        assert hf_backend_available() is True

    def test_false_when_torch_missing(self, monkeypatch):
        import importlib.util
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name: None if name == "torch" else object())
        assert hf_backend_available() is False

    def test_false_when_transformers_missing(self, monkeypatch):
        import importlib.util
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name: None if name == "transformers" else object())
        assert hf_backend_available() is False

    def test_false_when_probe_raises(self, monkeypatch):
        import importlib.util

        def boom(name):
            raise ValueError("half-installed namespace package")
        monkeypatch.setattr(importlib.util, "find_spec", boom)
        assert hf_backend_available() is False


# ------------------------------------------------------------------ #
#  HF VRAM size estimate (safetensors param count -> bf16 footprint)  #
# ------------------------------------------------------------------ #

class TestHfParamBytes:
    @pytest.mark.parametrize("st,expected", [
        ({"total": 134515008}, 269030016),                        # 134.5M * 2 (bf16)
        ({"total": 1_000_000_000, "parameters": {"BF16": 1_000_000_000}}, 2_000_000_000),
        (None, None),                                             # no metadata
        ({}, None),                                               # no total
        ({"total": 0}, None),                                    # zero -> unknown
        ({"total": -5}, None),                                   # negative -> unknown
        ({"total": "big"}, None),                               # non-int -> unknown
        ({"total": True}, None),                                # bool is not a count
    ])
    def test_param_bytes(self, st, expected):
        assert hf_param_bytes(st) == expected


class TestHfSearchSize:
    def test_hf_results_carry_size_and_request_expand(self, monkeypatch):
        import json as _json
        seen_urls = []

        def fake(url, **kw):
            seen_urls.append(url)
            payload = [
                {"id": "org/sized", "downloads": 3, "likes": 1, "lastModified": "",
                 "safetensors": {"total": 100}},
                {"id": "org/nometa", "downloads": 2, "likes": 0, "lastModified": ""},
            ]
            return url, "application/json", _json.dumps(payload).encode("utf-8")

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        results = hf_search("x", limit=5, formats=["hf"])
        # Repeated-key (doseq) encoding: distinct expand[] pairs, NOT a list repr.
        # Asserting a SECOND expand pair (downloads) proves doseq=True is in effect
        # (a non-doseq urlencode would emit one expand[]=['safetensors',...] blob).
        assert "expand%5B%5D=safetensors" in seen_urls[0]
        assert "expand%5B%5D=downloads" in seen_urls[0]
        by_id = {r["id"]: r for r in results}
        assert by_id["org/sized"]["size_bytes"] == 200    # 100 params * 2 bytes
        assert by_id["org/nometa"]["size_bytes"] is None  # unknown -> None, not 0

    def test_gguf_results_have_no_size_key(self, monkeypatch):
        _mock_fetch(monkeypatch, [{"id": "org/g", "downloads": 1}])
        results = hf_search("x", formats=["gguf"])
        assert "size_bytes" not in results[0]   # GGUF is sized per-file, not here

    def test_both_format_repo_keeps_hf_size(self, monkeypatch):
        """A repo in both formats enters via the gguf list (no size); the hf pass's
        size estimate must still be carried onto it."""
        import json as _json
        import urllib.parse

        gguf = [{"id": "org/both", "downloads": 9}]
        hf = [{"id": "org/both", "downloads": 9, "safetensors": {"total": 50}}]

        def fake(url, **kw):
            filt = dict(urllib.parse.parse_qsl(
                urllib.parse.urlparse(url).query)).get("filter")
            payload = gguf if filt == "gguf" else hf
            return url, "application/json", _json.dumps(payload).encode("utf-8")

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        results = hf_search("x", limit=5, formats=["gguf", "hf"])
        both = next(r for r in results if r["id"] == "org/both")
        assert sorted(both["formats"]) == ["gguf", "hf"]
        assert both["size_bytes"] == 100        # 50 params * 2, carried from hf pass


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


# ------------------------------------------------------------------ #
#  Multi-GPU: enumeration + main-GPU selection                        #
# ------------------------------------------------------------------ #

class TestListGpus:
    def _fake_torch(self, devices):
        """devices: [(name, free_bytes, total_bytes), ...]."""
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = len(devices)
        fake.cuda.get_device_name.side_effect = lambda i: devices[i][0]
        fake.cuda.mem_get_info.side_effect = lambda i: (devices[i][1], devices[i][2])
        return patch.dict(sys.modules, {"torch": fake})

    def test_enumerates_all_devices_with_names(self):
        devices = [("RTX 4090", 20_000_000_000, 24_000_000_000),
                   ("RTX 3060", 10_000_000_000, 12_000_000_000)]
        with self._fake_torch(devices):
            gpus = list_gpus()
        assert gpus == [
            {"index": 0, "name": "RTX 4090", "total": 24_000_000_000, "free": 20_000_000_000},
            {"index": 1, "name": "RTX 3060", "total": 12_000_000_000, "free": 10_000_000_000},
        ]

    def test_three_plus_devices_all_enumerated(self):
        devices = [(f"GPU{i}", i * 1_000_000_000, 8_000_000_000) for i in range(4)]
        with self._fake_torch(devices):
            gpus = list_gpus()
        assert len(gpus) == 4
        assert [g["index"] for g in gpus] == [0, 1, 2, 3]
        assert [g["name"] for g in gpus] == ["GPU0", "GPU1", "GPU2", "GPU3"]

    def test_falls_back_to_nvidia_smi_without_torch(self):
        csv = ("0, NVIDIA RTX 4090, 24576, 20000\n"
               "1, NVIDIA RTX 3060, 12288, 10000\n")
        fake_proc = MagicMock(returncode=0, stdout=csv)
        with patch.dict(sys.modules, {"torch": None}), \
             patch("subprocess.run", return_value=fake_proc):
            gpus = list_gpus()
        assert len(gpus) == 2
        assert gpus[0] == {"index": 0, "name": "NVIDIA RTX 4090",
                            "total": 24576 * 1024 ** 2, "free": 20000 * 1024 ** 2}
        assert gpus[1]["index"] == 1
        assert gpus[1]["name"] == "NVIDIA RTX 3060"

    def test_empty_when_nothing_available(self):
        with patch.dict(sys.modules, {"torch": None}), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            assert list_gpus() == []

    def test_empty_when_torch_sees_no_cuda(self):
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake}), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            assert list_gpus() == []


class TestResolveMainGpuIndex:
    def test_none_returns_zero_without_querying_devices(self, monkeypatch):
        calls = []

        def _tracked_list_gpus():
            calls.append(1)
            return []

        monkeypatch.setattr("localm.discover.list_gpus", _tracked_list_gpus)
        assert resolve_main_gpu_index(None) == 0
        assert calls == []

    def test_configured_zero_used_without_querying_devices(self, monkeypatch):
        calls = []

        def _tracked_list_gpus():
            calls.append(1)
            return []

        monkeypatch.setattr("localm.discover.list_gpus", _tracked_list_gpus)
        assert resolve_main_gpu_index(0) == 0
        assert calls == []

    def test_valid_index_within_range_is_used(self):
        gpus = [{"index": 0}, {"index": 1}, {"index": 2}]
        assert resolve_main_gpu_index(2, gpus=gpus) == 2

    def test_out_of_range_falls_back_to_zero_with_warning(self, caplog):
        gpus = [{"index": 0}, {"index": 1}]
        with caplog.at_level("WARNING", logger="localm"):
            idx = resolve_main_gpu_index(5, gpus=gpus)
        assert idx == 0
        assert any("main_gpu_index" in r.message for r in caplog.records)

    def test_negative_index_falls_back_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            idx = resolve_main_gpu_index(-1)
        assert idx == 0
        assert any("main_gpu_index" in r.message for r in caplog.records)

    def test_non_integer_falls_back_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            idx = resolve_main_gpu_index("not-a-number")
        assert idx == 0
        assert any("main_gpu_index" in r.message for r in caplog.records)

    def test_unmeasurable_passes_through_unchecked(self, monkeypatch):
        # gpus not injected and list_gpus() finds nothing (no torch, no
        # nvidia-smi): the configured index cannot be cross-checked, so it is
        # trusted rather than discarded (documented boundary).
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [])
        assert resolve_main_gpu_index(3) == 3


class TestApplyMainGpu:
    def test_unset_leaves_native_default_untouched(self):
        mp = SimpleNamespace(main_gpu=0)
        apply_main_gpu(mp, config={"main_gpu_index": None})
        assert mp.main_gpu == 0

    def test_configured_index_is_set_on_mp(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: [{"index": 0}, {"index": 1}])
        mp = SimpleNamespace(main_gpu=0)
        apply_main_gpu(mp, config={"main_gpu_index": 1})
        assert mp.main_gpu == 1

    def test_invalid_configured_index_falls_back_to_zero_with_warning(self, monkeypatch, caplog):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        mp = SimpleNamespace(main_gpu=99)
        with caplog.at_level("WARNING", logger="localm"):
            apply_main_gpu(mp, config={"main_gpu_index": 7})
        assert mp.main_gpu == 0
        assert any("main_gpu_index" in r.message for r in caplog.records)

    def test_reads_load_config_when_config_not_passed(self, monkeypatch):
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        mp = SimpleNamespace(main_gpu=0)
        apply_main_gpu(mp)
        assert mp.main_gpu == 0


class TestVramInfoRespectsConfiguredDevice:
    """vram_info() must reflect the CONFIGURED main GPU, not always device 0,
    once list_gpus() can see more than one device."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000},
        {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000},
    ]

    def test_uses_configured_device(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 1})
        assert vram_info() == {"total": 12_000_000_000, "free": 10_000_000_000}

    def test_defaults_to_device_zero_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": None})
        assert vram_info() == {"total": 24_000_000_000, "free": 20_000_000_000}

    def test_invalid_configured_index_falls_back_to_zero(self, monkeypatch, caplog):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS[:1])
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"main_gpu_index": 9})
        with caplog.at_level("WARNING", logger="localm"):
            info = vram_info()
        assert info == {"total": 24_000_000_000, "free": 20_000_000_000}
