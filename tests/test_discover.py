# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for localm.discover - HF search, quant parsing, VRAM fit badges.
All HuggingFace calls are mocked; no real network."""

import ctypes
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm import discover
# Imported at collection time so TestListGpus's patch.dict(sys.modules, {"torch":
# ...}) has nothing to evict. discover imports gpu_usage lazily (inside
# _apply_device_global_free); an eviction would leave a stale module object bound
# on the localm package, which is what monkeypatch's "localm.gpu_usage.X" string
# targets resolve against, while the code under test re-imports a fresh one.
from localm import gpu_usage as _gpu_usage_imported_early  # noqa: F401
from localm.discover import (
    DiscoverError, GPU_PROBE_BUSY, GPU_PROBE_INCONCLUSIVE, GPU_PROBE_OK,
    GPU_PROBE_TIMEOUT,
    _GPU_PROBE_CLI_DEADLINE, _GPU_PROBE_DEADLINE, _LLAMA_SPLIT_MODE_LAYER,
    _MAX_GPU_SPLIT_INDEX, _TENSOR_SPLIT_FALLBACK_CAPACITY,
    _moe_signal, _native_backend_has_vulkan,
    _quant_of, applied_split_device_count, apply_gpu_split, apply_main_gpu,
    classify_hf_metadata, fit_label, gpu_split_shortfall,
    hf_backend_available, hf_gguf_files, hf_param_bytes, hf_search, list_gpus,
    resolve_gpu_split, resolve_main_gpu_index, split_device_count, vram_capacity,
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
        ("qwen-agentworld-35b-a3b-mxfp4_moe.gguf", "MXFP4_MOE"),   # real filename
        ("model-MXFP4.gguf", "MXFP4"),
        ("model-TQ1_0.gguf", "TQ1_0"),
        ("model-TQ2_0.gguf", "TQ2_0"),
    ])
    def test_quant_of(self, name, quant):
        assert _quant_of(name) == quant

    def test_quant_of_prefers_mxfp4_over_an_earlier_base_precision_token(self):
        """Real filename (elbelga/Huihui-Qwen3.6-35B-A3B-abliterated_MXFP4_MOE):
        the non-expert-tensor precision is named FIRST, so plain re.search (which
        returns the LEFTMOST match) would report 'BF16'/'F16' - the un-quantized
        dtype of the tensors that are NOT the point - instead of the actual MoE
        expert quantization."""
        assert _quant_of(
            "Huihui-Qwen3.6-35B-A3B-abliterated-bf16_MXFP4_MOE.gguf") == "MXFP4_MOE"
        assert _quant_of(
            "Huihui-Qwen3.6-35B-A3B-abliterated-f16_MXFP4_MOE.gguf") == "MXFP4_MOE"


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
        with pytest.raises(DiscoverError, match="net_mode=off") as ei:
            hf_search("anything")
        assert ei.value.off is True

    def test_net_off_but_downloads_allowed_proceeds(self, monkeypatch):
        """net_allow_model_downloads exempts discovery (a user-initiated
        prelude to a pull) from the off floor, same as an explicit pull."""
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        monkeypatch.setattr(
            "localm.netpolicy.downloads_allowed_when_off", lambda: True)
        _mock_fetch(monkeypatch, [
            {"id": "org/g1", "tags": ["gguf"], "downloads": 5, "likes": 0},
        ])
        results = hf_search("x", formats=["gguf"])
        assert [r["id"] for r in results] == ["org/g1"]

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


class TestHfAuthHeaders:
    """_hf_auth_headers is the tiny pure helper HFSource (sources.py) relies
    on to turn a configured token into HF's own Authorization convention."""

    def test_none_when_no_token(self):
        assert discover._hf_auth_headers(None) is None
        assert discover._hf_auth_headers("") is None

    def test_bearer_header_when_token_set(self):
        assert discover._hf_auth_headers("hf_abc123") == {
            "Authorization": "Bearer hf_abc123"}


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


class TestGgufClassifyExpand:
    """A classified (model_types-scoped) gguf-format query needs its own
    expand[] fields: gguf.architecture for classify_hf_metadata, plus
    downloads/likes/lastModified, which `expand` silently drops the moment ANY
    field is requested - a classified gguf query requesting only
    pipeline_tag/library_name/tags comes back with downloads and likes ABSENT,
    not merely zero, where they are present with no expand at all."""

    def test_classified_gguf_query_requests_stats_and_architecture_fields(self, monkeypatch):
        # Capture the raw URL (not parse_qsl's `seen` dict, which collapses
        # repeated expand[] keys to the last one) to assert every field was
        # requested.
        urls = _urls_capture(monkeypatch, [{
            "id": "mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF",
            "downloads": 9970, "likes": 20, "lastModified": "2026-05-21T21:35:06.000Z",
            "tags": ["gguf", "conversational"],
            "gguf": {"architecture": "qwen35moe", "total": 35505251456},
        }])
        results = hf_search("carnice", limit=5, formats=["gguf"], model_types=["llm"])
        joined = urls[0]
        for field in ("downloads", "likes", "lastModified", "gguf", "config",
                      "pipeline_tag", "library_name", "tags"):
            assert f"expand%5B%5D={field}" in joined, f"{field} missing from {joined}"
        assert results[0]["downloads"] == 9970
        assert results[0]["likes"] == 20
        assert results[0]["updated"] == "2026-05-21T21:35:06.000Z"

    def test_hf_format_classify_does_not_request_gguf_expand(self, monkeypatch):
        """The gguf expand field is meaningless for an hf/safetensors result (it
        has no GGUF header) - only requested on the gguf format branch."""
        urls = _urls_capture(monkeypatch, [
            {"id": "org/hf-repo", "downloads": 1,
             "pipeline_tag": "text-generation", "library_name": "transformers",
             "tags": ["transformers", "text-generation"]}])
        hf_search("x", limit=5, formats=["hf"], model_types=["llm"])
        assert "expand%5B%5D=gguf" not in urls[0]
        assert "expand%5B%5D=config" in urls[0]

    def test_hf_format_classify_keeps_its_own_stats_and_size_expand(self, monkeypatch):
        """Guards the if/elif split above: fmt=='hf' must still get its own
        safetensors+downloads+likes+lastModified expand when classify is ALSO
        on - the new `elif classify` branch (gguf-only) must never suppress the
        pre-existing, unrelated `if fmt == 'hf'` branch."""
        urls = _urls_capture(monkeypatch, [
            {"id": "org/hf-repo", "downloads": 1, "safetensors": {"total": 10},
             "pipeline_tag": "text-generation", "library_name": "transformers",
             "tags": ["transformers", "text-generation"]}])
        hf_search("x", limit=5, formats=["hf"], model_types=["llm"])
        for field in ("safetensors", "downloads", "likes", "lastModified"):
            assert f"expand%5B%5D={field}" in urls[0], f"{field} missing from {urls[0]}"

    def test_end_to_end_real_repo_shape_resolves_to_llm_via_hf_search(self, monkeypatch):
        """Integration proof (not just the pure classify_hf_metadata unit test)
        that the wiring from HF's raw JSON shape all the way to the row's
        detected_type badge is correct: mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-GGUF
        real metadata - no pipeline_tag - resolves through hf_search itself."""
        _mock_fetch(monkeypatch, [{
            "id": "mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-GGUF",
            "downloads": 100, "likes": 5,
            "tags": ["gguf", "quantized", "apex", "moe", "mixture-of-experts",
                     "qwen3", "carnice", "agentic", "tool-calling",
                     "license:apache-2.0", "endpoints_compatible", "region:us",
                     "conversational"],
            "gguf": {"architecture": "qwen35moe", "total": 34660610688},
        }])
        results = hf_search("carnice", limit=5, formats=["gguf"], model_types=["llm"])
        assert results[0]["detected_type"] == "llm"

    def test_end_to_end_embedding_architecture_via_hf_search(self, monkeypatch):
        """A gguf result with a verified embedding architecture and NO other
        classifying signal resolves to 'embedding' through the full hf_search
        path, not just the pure function."""
        _mock_fetch(monkeypatch, [{
            "id": "org/bert-embed-gguf", "downloads": 3, "tags": ["gguf"],
            "gguf": {"architecture": "nomic-bert-moe", "total": 1},
        }])
        results = hf_search("x", limit=5, formats=["gguf"], model_types=["embedding"])
        assert results[0]["detected_type"] == "embedding"

    def test_malformed_gguf_and_config_expand_fields_degrade_not_crash(self, monkeypatch):
        """A non-dict truthy value for the gguf/config expand field (a malformed
        or adversarial API response - HF's documented shape is always an object
        or absent) must degrade this row's architecture signal to None, matching
        hf_param_bytes' isinstance guard on safetensors a few lines above - not
        raise and take down every OTHER row in the same query with it."""
        _mock_fetch(monkeypatch, [
            {"id": "org/malformed-gguf", "downloads": 1, "tags": ["gguf"],
             "gguf": "not-a-dict", "config": ["also", "not", "a", "dict"]},
            {"id": "org/fine-gguf", "downloads": 2,
             "tags": ["gguf", "conversational"]},
        ])
        results = hf_search("x", limit=5, formats=["gguf"], model_types=["llm"])
        by_id = {r["id"]: r for r in results}
        assert by_id["org/malformed-gguf"]["detected_type"] == "unknown"
        assert by_id["org/fine-gguf"]["detected_type"] == "llm"


class TestArchitectureMoeParamCount:
    """architecture/moe/param_count: the display-only fields a classified search
    result carries about WHAT the model is, layered on classify_hf_metadata.
    gguf.total tracks the model's total parameter count (stable across quants of
    the same repo, not a byte size), and architecture-contains-"moe" is reliable
    but NOT exhaustive - TheBloke/Mixtral-8x7B-v0.1-GGUF (a real MoE) reports
    architecture=="llama"."""

    def test_confirmed_moe_from_architecture_string(self, monkeypatch):
        _mock_fetch(monkeypatch, [{
            "id": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "downloads": 1,
            "tags": ["gguf", "conversational"],
            "gguf": {"architecture": "qwen3moe", "total": 30532122624},
        }])
        r = hf_search("x", limit=5, formats=["gguf"], model_types=["llm"])[0]
        assert r["architecture"] == "qwen3moe"
        assert r["moe"] == "confirmed"
        assert r["param_count"] == 30532122624

    def test_likely_moe_from_repo_name_when_architecture_does_not_say_so(self, monkeypatch):
        """A counter-example: this real Mixtral repo's architecture reports
        'llama', not 'mixtral' or anything containing 'moe' - an older
        conversion predating the dedicated arch tag. The name-pattern fallback
        must catch it, and must NOT claim it as confirmed."""
        _mock_fetch(monkeypatch, [{
            "id": "TheBloke/Mixtral-8x7B-v0.1-GGUF", "downloads": 1,
            "tags": ["gguf"],
            "gguf": {"architecture": "llama", "total": 46702792704},
        }])
        r = hf_search("x", limit=5, formats=["gguf"], model_types=["llm"])[0]
        assert r["architecture"] == "llama"
        assert r["moe"] == "likely"

    def test_no_moe_evidence_is_none_not_a_false_dense_claim(self, monkeypatch):
        """Absence of both signals must abstain (None), never assert the model
        is dense - same discipline classify_hf_metadata already applies to type."""
        _mock_fetch(monkeypatch, [{
            "id": "meta-llama/Llama-3.1-8B-Instruct-GGUF", "downloads": 1,
            "tags": ["gguf", "conversational"],
            "gguf": {"architecture": "llama", "total": 8030261248},
        }])
        r = hf_search("x", limit=5, formats=["gguf"], model_types=["llm"])[0]
        assert r["moe"] is None

    def test_moe_name_patterns_match_case_insensitively(self):
        assert _moe_signal(None, "org/Some-MoE-Model") == "likely"
        assert _moe_signal(None, "org/Mixtral-8X7B-Instruct") == "likely"
        assert _moe_signal(None, "org/Qwen3-30B-A3B") == "likely"
        assert _moe_signal(None, "org/Llama-3.2-1B-Instruct") is None

    def test_param_count_from_hf_format_safetensors_total(self, monkeypatch):
        _mock_fetch(monkeypatch, [{
            "id": "org/hf-repo", "downloads": 1,
            "pipeline_tag": "text-generation", "library_name": "transformers",
            "tags": ["transformers", "text-generation"],
            "safetensors": {"total": 8_000_000_000},
        }])
        r = hf_search("x", limit=5, formats=["hf"], model_types=["llm"])[0]
        assert r["param_count"] == 8_000_000_000

    def test_malformed_gguf_total_degrades_to_none_not_crash(self, monkeypatch):
        """Same isinstance/positive guard as hf_param_bytes: a malformed or
        adversarial 'total' (string, negative, bool) must degrade this row's
        param_count to None, never raise or misreport a bool as a count."""
        _mock_fetch(monkeypatch, [
            {"id": "org/string-total", "downloads": 1, "tags": ["gguf"],
             "gguf": {"architecture": "llama", "total": "not-a-number"}},
            {"id": "org/negative-total", "downloads": 1, "tags": ["gguf"],
             "gguf": {"architecture": "llama", "total": -5}},
            {"id": "org/bool-total", "downloads": 1, "tags": ["gguf"],
             "gguf": {"architecture": "llama", "total": True}},
            {"id": "org/fine-total", "downloads": 2, "tags": ["gguf"],
             "gguf": {"architecture": "llama", "total": 7_000_000_000}},
        ])
        results = hf_search("x", limit=5, formats=["gguf"], model_types=["llm"])
        by_id = {r["id"]: r for r in results}
        assert by_id["org/string-total"]["param_count"] is None
        assert by_id["org/negative-total"]["param_count"] is None
        assert by_id["org/bool-total"]["param_count"] is None
        assert by_id["org/fine-total"]["param_count"] == 7_000_000_000

    def test_untyped_legacy_search_carries_none_of_the_new_fields(self, monkeypatch):
        """model_types=None (the CLI `localm search` / MCP path) never requests
        the gguf/config expand fields, so the new fields must be entirely
        absent - byte-for-byte the pre-existing response shape, same contract
        detected_type already holds."""
        _mock_fetch(monkeypatch, [{"id": "org/repo", "downloads": 1, "tags": ["gguf"]}])
        r = hf_search("x", limit=5, formats=["gguf"])[0]
        assert "architecture" not in r
        assert "moe" not in r
        assert "param_count" not in r


# ------------------------------------------------------------------ #
#  classify_hf_metadata - pure classification, no network              #
# ------------------------------------------------------------------ #

class TestClassifyHfMetadata:
    """Fixtures below are REAL repo metadata from the HF API, not synthesized.
    vae/text-encoder canonical repos carry none of the signals this function
    looks for - see discover.py's _HF_TYPE_FILTER_DEFAULT comment."""

    def test_llm(self):
        assert classify_hf_metadata("text-generation", "transformers",
                                     ["transformers", "text-generation"]) == "llm"

    def test_embedding_bge_m3(self):
        # BAAI/bge-m3
        assert classify_hf_metadata(
            "sentence-similarity", "sentence-transformers",
            ["sentence-transformers", "pytorch", "xlm-roberta",
             "feature-extraction", "sentence-similarity"]) == "embedding"

    def test_diffusion_flux_dev(self):
        # black-forest-labs/FLUX.1-dev
        assert classify_hf_metadata(
            "text-to-image", "diffusers",
            ["diffusers", "safetensors", "text-to-image", "flux"]) == "diffusion-unet"

    def test_lora_precedence_over_diffusion_pipeline_tag(self):
        """XLabs-AI/flux-RealismLora carries pipeline_tag=text-to-image
        (inherited from its FLUX base model) AND an exact 'lora' tag. The tag
        must win: checking the diffusion pipeline_tag first would misclassify
        this, and every other diffusion LoRA, as diffusion-unet."""
        assert classify_hf_metadata(
            "text-to-image", "diffusers",
            ["diffusers", "lora", "Stable Diffusion", "image-generation",
             "Flux", "text-to-image",
             "base_model:adapter:black-forest-labs/FLUX.1-dev"]) == "lora"

    def test_lora_via_peft_library(self):
        assert classify_hf_metadata("text-generation", "peft",
                                     ["peft", "safetensors", "text-generation"]) == "lora"

    def test_vae_sd_vae_ft_mse_has_no_classifying_metadata(self):
        # stabilityai/sd-vae-ft-mse - the canonical SD VAE. No "vae" tag, no
        # pipeline_tag at all. Cannot be classified from hard metadata - must
        # come back 'unknown', not a wrong guess.
        assert classify_hf_metadata(
            None, "diffusers",
            ["diffusers", "safetensors", "stable-diffusion",
             "stable-diffusion-diffusers"]) == "unknown"

    def test_vae_with_explicit_tag(self):
        assert classify_hf_metadata(
            "image-to-image", "diffusers", ["diffusers", "vae"]) == "vae"

    def test_text_encoder_flux_text_encoders_has_no_classifying_metadata(self):
        # comfyanonymous/flux_text_encoders - the standard FLUX text encoder
        # repo used throughout the ComfyUI ecosystem. No pipeline_tag, no
        # library_name, no tag beyond a license marker.
        assert classify_hf_metadata(None, None, ["license:apache-2.0", "region:us"]) == "unknown"

    def test_text_encoder_with_explicit_tag(self):
        assert classify_hf_metadata(None, None, ["text-encoder"]) == "text-encoder"
        assert classify_hf_metadata(None, None, ["clip"]) == "text-encoder"

    def test_substring_is_not_a_match(self):
        """A tag that merely CONTAINS 'lora'/'vae'/'clip' must not misclassify
        (e.g. an 'exploration' tag)."""
        assert classify_hf_metadata(None, None, ["exploration", "clipboard-app"]) == "unknown"

    def test_nothing_resolves_returns_unknown_not_llm(self):
        assert classify_hf_metadata(None, None, []) == "unknown"
        assert classify_hf_metadata("image-classification", "timm", ["timm"]) == "unknown"

    # -------------------------------------------------------------- #
    # architecture param (gguf.architecture / config.model_type) and  #
    # the tag-set LLM/embedding fallback                              #
    # -------------------------------------------------------------- #

    def test_embedding_architecture_signal(self):
        """A GGUF whose own header architecture is a verified embedding-only
        llama.cpp architecture is 'embedding' even with zero other metadata -
        the strongest signal this function has, since it comes from the
        model's own file rather than a repo author's tags."""
        assert classify_hf_metadata(None, None, [], "bert") == "embedding"
        assert classify_hf_metadata(None, None, [], "nomic-bert-moe") == "embedding"

    def test_architecture_signal_is_case_insensitive_and_optional(self):
        assert classify_hf_metadata(None, None, [], "BERT") == "embedding"
        assert classify_hf_metadata(None, None, [], None) == "unknown"
        assert classify_hf_metadata(None, None, []) == "unknown"   # default arg

    def test_architecture_unrecognised_falls_through_not_wrong_guess(self):
        """A causal-decoder architecture (llama/qwen35moe/deepseek2/mixtral) is
        NOT in the embedding allowlist, and classify_hf_metadata has no positive
        LLM-by-architecture list (too broad/unverifiable to maintain) - it must
        fall through to the pipeline_tag/tagset checks, never guess 'llm' from
        architecture alone."""
        assert classify_hf_metadata(None, None, [], "qwen35moe") == "unknown"

    def test_lora_tag_wins_over_embedding_architecture(self):
        """Precedence: the exact-tag checks run BEFORE the architecture check,
        same reasoning as the diffusion-vs-lora precedence test above - a LoRA
        adapter for an embedding base model is still a lora, not a full
        embedding checkpoint."""
        assert classify_hf_metadata(
            None, "peft", ["peft", "lora"], "bert") == "lora"

    def test_llm_tagset_conversational_without_pipeline_tag(self):
        """The exact shape of a HF-repacked GGUF-only upload: pipeline_tag is
        entirely absent (never set on the quantizer's own repo), but the base
        model's 'conversational' tag survives onto it."""
        assert classify_hf_metadata(
            None, None, ["gguf", "conversational"]) == "llm"

    def test_llm_tagset_text_generation_without_pipeline_tag(self):
        assert classify_hf_metadata(
            None, None, ["gguf", "text-generation"]) == "llm"

    def test_embedding_tagset_without_pipeline_tag(self):
        assert classify_hf_metadata(
            None, None, ["gguf", "feature-extraction"]) == "embedding"

    def test_image_text_to_text_is_llm_not_diffusion(self):
        """A vision-language chat model (e.g. a Qwen-VL GGUF) declares this
        pipeline_tag - it is still an LLM (a chat model with image input), never
        routed to diffusion-unet."""
        assert classify_hf_metadata(
            "image-text-to-text", "transformers",
            ["transformers", "gguf", "image-text-to-text"]) == "llm"

    # Real repo metadata captured from the HF API for 8 repos that carry NO
    # pipeline_tag (a HF-repacked GGUF-only upload commonly never sets it); the
    # tag-set 'conversational' fallback and/or the gguf.architecture signal
    # resolve all 8 to 'llm'.

    def test_real_repo_mudler_carnice_apex_mtp(self):
        # mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF
        assert classify_hf_metadata(
            None, None,
            ["gguf", "quantized", "apex", "apex-mtp", "moe", "mixture-of-experts",
             "qwen3", "qwen3.6", "speculative-decoding", "self-speculative", "mtp",
             "base_model:samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B",
             "base_model:quantized:samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B",
             "license:apache-2.0", "endpoints_compatible", "region:us",
             "conversational"],
            "qwen35moe") == "llm"

    def test_real_repo_mudler_carnice_apex(self):
        # mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-GGUF
        assert classify_hf_metadata(
            None, None,
            ["gguf", "quantized", "apex", "moe", "mixture-of-experts", "qwen3",
             "carnice", "agentic", "tool-calling",
             "base_model:samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B",
             "base_model:quantized:samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B",
             "license:apache-2.0", "endpoints_compatible", "region:us",
             "conversational"],
            "qwen35moe") == "llm"

    def test_real_repo_thebloke_mixtral_moe_rp_story(self):
        # TheBloke/Mixtral-8x7B-MoE-RP-Story-GGUF - library_name=transformers,
        # gguf.architecture=llama (Mixtral is llama.cpp's llama arch family).
        assert classify_hf_metadata(
            None, "transformers",
            ["transformers", "gguf", "mixtral", "not-for-all-audiences", "nsfw",
             "base_model:Undi95/Mixtral-8x7B-MoE-RP-Story",
             "base_model:quantized:Undi95/Mixtral-8x7B-MoE-RP-Story",
             "license:cc-by-nc-4.0", "region:us", "conversational"],
            "llama") == "llm"

    def test_real_repo_mradermacher_carnice_i1(self):
        # mradermacher/Carnice-Qwen3.6-MoE-35B-A3B-i1-GGUF
        assert classify_hf_metadata(
            None, "transformers",
            ["transformers", "gguf", "qwen3.6", "moe", "hermes", "agentic",
             "tool-calling", "qlora", "unsloth", "carnice", "en",
             "dataset:bespokelabs/Bespoke-Stratos-17k",
             "dataset:AI-MO/NuminaMath-CoT",
             "dataset:kai-os/carnice-glm5-hermes-traces",
             "dataset:open-thoughts/OpenThoughts-Agent-v1-SFT",
             "base_model:samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B",
             "base_model:quantized:samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B",
             "license:apache-2.0", "endpoints_compatible", "region:us", "imatrix",
             "conversational"],
            "qwen35moe") == "llm"

    def test_real_repo_mradermacher_droplychee_moe_v2(self):
        # mradermacher/droplychee-moe-v2-i1-GGUF - gguf.architecture=deepseek2.
        assert classify_hf_metadata(
            None, "transformers",
            ["transformers", "gguf", "text-generation-inference", "unsloth",
             "glm4_moe_lite", "en", "base_model:droplychee/droplychee-moe-v2",
             "base_model:quantized:droplychee/droplychee-moe-v2",
             "license:apache-2.0", "endpoints_compatible", "region:us", "imatrix",
             "conversational"],
            "deepseek2") == "llm"

    def test_real_repo_elbelga_huihui_qwen_abliterated_mxfp4(self):
        # elbelga/Huihui-Qwen3.6-35B-A3B-abliterated_MXFP4_MOE - the ONLY one of
        # the 8 with an explicit pipeline_tag: image-text-to-text (VLM).
        assert classify_hf_metadata(
            "image-text-to-text", "transformers",
            ["transformers", "gguf", "abliterated", "uncensored",
             "image-text-to-text",
             "base_model:Qwen/Qwen3.6-35B-A3B",
             "base_model:quantized:Qwen/Qwen3.6-35B-A3B", "license:apache-2.0",
             "endpoints_compatible", "region:us", "imatrix", "conversational"],
            "qwen35moe") == "llm"

    def test_real_repo_freedomaisvr_qwen_agentworld_mxfp4_moe(self):
        # FreedomAISVR/Qwen-AgentWorld-35B-A3B-MXFP4-MOE-GGUF
        assert classify_hf_metadata(
            None, None,
            ["gguf", "qwen", "qwen3.5", "moe", "agent", "world-model",
             "mxfp4_moe", "vision", "multimodal", "35b", "en", "multilingual",
             "base_model:Qwen/Qwen-AgentWorld-35B-A3B",
             "base_model:quantized:Qwen/Qwen-AgentWorld-35B-A3B",
             "license:apache-2.0", "endpoints_compatible", "region:us",
             "conversational"],
            "qwen35moe") == "llm"

    def test_real_repo_unsloth_cogito_v2_deepseek_moe(self):
        # unsloth/cogito-v2-preview-deepseek-671B-MoE-GGUF
        assert classify_hf_metadata(
            None, "transformers",
            ["transformers", "gguf", "unsloth",
             "base_model:deepcogito/cogito-v2-preview-deepseek-671B-MoE",
             "base_model:quantized:deepcogito/cogito-v2-preview-deepseek-671B-MoE",
             "license:mit", "endpoints_compatible", "region:us", "imatrix",
             "conversational"],
            "deepseek2") == "llm"


# ------------------------------------------------------------------ #
#  Type-scoped search (per-tab HF discovery)                           #
# ------------------------------------------------------------------ #

def _urls_capture(monkeypatch, payload):
    """Capture every request URL (repeated filter= keys survive, unlike a
    parse_qsl dict) and return *payload* for each call."""
    import json as _json
    urls: list[str] = []

    def fake(url, **kw):
        urls.append(url)
        return url, "application/json", _json.dumps(payload).encode("utf-8")
    monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
    return urls


class TestTypeScopedSearch:
    """Two orthogonal, independently-selectable axes: model TYPE (via
    model_types) and file FORMAT (gguf / hf==safetensors). Every filter value
    below matches the real HF API. See hf_search's docstring and
    _type_fmt_filter / _HF_TYPE_FILTER."""

    def test_embedding_hf_uses_feature_extraction_pipeline_tag(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "BAAI/bge-m3", "downloads": 20, "likes": 5,
             "pipeline_tag": "sentence-similarity", "library_name": "sentence-transformers",
             "tags": ["sentence-transformers", "feature-extraction", "sentence-similarity"]},
        ], seen)
        results = hf_search("bge", limit=5, formats=["hf"], model_types=["embedding"])
        assert seen["pipeline_tag"] == "feature-extraction"
        assert "filter" not in seen   # replaces the generic transformers filter entirely
        assert results[0]["detected_type"] == "embedding"

    def test_embedding_gguf_stays_plain_gguf_filter(self, monkeypatch):
        """No reliable gguf+embedding combined filter exists - conservative
        default: plain gguf format filter, query text does the narrowing, still
        classified for a badge when metadata is present."""
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "unsloth/bge-small-en-v1.5-GGUF", "downloads": 9,
             "pipeline_tag": "sentence-similarity", "tags": ["gguf", "feature-extraction"]},
        ], seen)
        results = hf_search("bge", limit=5, formats=["gguf"], model_types=["embedding"])
        assert seen["filter"] == "gguf"
        assert results[0]["detected_type"] == "embedding"

    def test_diffusion_hf_uses_text_to_image_pipeline_tag(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "black-forest-labs/FLUX.1-dev", "downloads": 500,
             "pipeline_tag": "text-to-image", "library_name": "diffusers",
             "tags": ["diffusers", "safetensors", "text-to-image", "flux"]},
        ], seen)
        results = hf_search("flux", limit=5, formats=["hf"], model_types=["diffusion-unet"])
        assert seen["pipeline_tag"] == "text-to-image"
        assert "filter" not in seen
        assert results[0]["detected_type"] == "diffusion-unet"

    def test_diffusion_gguf_adds_diffusers_filter_alongside_gguf(self, monkeypatch):
        """Repeated filter= keys (doseq-encoded AND):
        filter=diffusers&filter=gguf returns real GGUF diffusion repos. Asserted
        via the raw URL (parse_qsl collapses repeated keys to the last one)."""
        urls = _urls_capture(monkeypatch, [
            {"id": "wikeeyang/Krea2-Turbo-HD-V1", "downloads": 35,
             "pipeline_tag": "text-to-image", "tags": ["diffusers", "gguf"]}])
        results = hf_search("flux", limit=5, formats=["gguf"], model_types=["diffusion-unet"])
        assert "filter=gguf" in urls[0]
        assert "filter=diffusers" in urls[0]
        assert results[0]["detected_type"] == "diffusion-unet"

    def test_lora_hf_uses_peft_filter(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "XLabs-AI/flux-RealismLora", "downloads": 12,
             "pipeline_tag": "text-to-image", "library_name": "diffusers",
             "tags": ["diffusers", "lora", "text-to-image"]},
        ], seen)
        results = hf_search("flux realism", limit=5, formats=["hf"], model_types=["lora"])
        assert seen["filter"] == "peft"
        assert results[0]["detected_type"] == "lora"   # tag wins over pipeline_tag

    def test_lora_gguf_uses_gguf_format_filter(self, monkeypatch):
        """LoRA now honors the format axis (it no longer ignores it): the gguf
        side is the plain gguf format filter (no reliable gguf-LoRA narrowing;
        query text + the badge do the rest)."""
        seen = {}
        _mock_fetch(monkeypatch, [{"id": "some/gguf-lora", "downloads": 3,
                                   "tags": ["gguf", "lora"]}], seen)
        hf_search("lora", limit=5, formats=["gguf"], model_types=["lora"])
        assert seen["filter"] == "gguf"

    def test_vae_hf_uses_safetensors_format_filter(self, monkeypatch):
        """HF has no reliable VAE *type* filter (sd-vae-ft-mse carries no vae
        tag), but the *format* axis IS reliable: filter=safetensors returns
        exactly the canonical diffusers VAEs. So the vae tab's hf side narrows by
        format, and the badge stays honestly 'unknown'."""
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "stabilityai/sd-vae-ft-mse", "downloads": 900, "library_name": "diffusers",
             "tags": ["diffusers", "safetensors", "stable-diffusion", "stable-diffusion-diffusers"]},
        ], seen)
        results = hf_search("vae", limit=5, formats=["hf"], model_types=["vae"])
        assert seen["filter"] == "safetensors"
        assert results[0]["id"] == "stabilityai/sd-vae-ft-mse"
        assert results[0]["detected_type"] == "unknown"   # never a wrong guess

    def test_vae_gguf_uses_gguf_format_filter(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [{"id": "calcuis/pig-vae", "downloads": 5,
                                   "tags": ["gguf"]}], seen)
        hf_search("vae", limit=5, formats=["gguf"], model_types=["vae"])
        assert seen["filter"] == "gguf"

    def test_text_encoder_hf_uses_safetensors_format_filter(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "comfyanonymous/flux_text_encoders", "downloads": 400,
             "tags": ["license:apache-2.0", "region:us"]},
        ], seen)
        results = hf_search("text encoder", limit=5, formats=["hf"], model_types=["text-encoder"])
        assert seen["filter"] == "safetensors"
        assert results[0]["id"] == "comfyanonymous/flux_text_encoders"
        assert results[0]["detected_type"] == "unknown"

    def test_unknown_hf_uses_safetensors_format_filter(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [{"id": "madebyollin/taesd", "downloads": 50,
                                   "tags": ["diffusers", "safetensors"]}], seen)
        hf_search("taesd", limit=5, formats=["hf"], model_types=["unknown"])
        assert seen["filter"] == "safetensors"

    def test_llm_hf_keeps_transformers_filter(self, monkeypatch):
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "meta-llama/Llama-3.2-1B-Instruct", "downloads": 1,
             "pipeline_tag": "text-generation", "library_name": "transformers",
             "tags": ["transformers", "text-generation"]},
        ], seen)
        results = hf_search("llama", formats=["hf"], model_types=["llm"])
        assert seen["filter"] == "transformers"
        assert results[0]["detected_type"] == "llm"

    def test_singular_model_type_is_alias_for_one_element_list(self, monkeypatch):
        """Back-compat: the route/tests still pass model_type= (singular). It is
        exactly model_types=[that]."""
        seen = {}
        _mock_fetch(monkeypatch, [
            {"id": "BAAI/bge-m3", "downloads": 1,
             "pipeline_tag": "feature-extraction", "tags": ["sentence-transformers"]}], seen)
        results = hf_search("bge", formats=["hf"], model_type="embedding")
        assert seen["pipeline_tag"] == "feature-extraction"
        assert results[0]["detected_type"] == "embedding"

    def test_default_no_type_has_no_detected_type_key(self, monkeypatch):
        """Back-compat guard: today's ONLY real external contract - the CLI's
        `localm search` and the MCP search_models tool, neither of which passes
        a type. Must be byte-for-byte what shipped before type-scoped search:
        the hf side is filter=transformers, no classify, no detected_type."""
        seen = {}
        _mock_fetch(monkeypatch, [{"id": "org/g", "downloads": 1}], seen)
        results = hf_search("x", formats=["gguf"])   # no model_type/model_types
        assert "detected_type" not in results[0]
        assert "pipeline_tag" not in seen and "expand[]" not in seen
        assert seen["filter"] == "gguf"

    def test_default_no_type_hf_side_stays_transformers(self, monkeypatch):
        """The legacy (CLI/MCP) hf side must remain filter=transformers, NOT the
        new filter=safetensors - the byte-compat contract is unchanged."""
        seen = {}
        _mock_fetch(monkeypatch, [{"id": "org/hf", "downloads": 1}], seen)
        hf_search("x", formats=["hf"])
        assert seen["filter"] == "transformers"

    def test_all_types_selected_collapses_to_broad_format_filters(self, monkeypatch):
        """Selecting every type is 'search everything', not a 7*fmt fan-out: the
        widest reliable format filter (gguf / safetensors), classified. Two
        queries, not fourteen - and safetensors (not transformers) so diffusion/
        vae/encoders are NOT excluded from the non-gguf side."""
        urls = _urls_capture(monkeypatch, [{"id": "org/x", "downloads": 1, "tags": ["gguf"]}])
        all_types = ["llm", "embedding", "diffusion-unet", "text-encoder",
                     "vae", "lora", "unknown"]
        hf_search("x", limit=5, formats=["gguf", "hf"], model_types=all_types)
        assert len(urls) == 2                         # one per format, not 14
        joined = " ".join(urls)
        assert "filter=gguf" in joined
        assert "filter=safetensors" in joined
        assert "filter=transformers" not in joined    # broad hf side is safetensors

    def test_multi_type_dedupes_identical_resolved_queries(self, monkeypatch):
        """vae + text-encoder both resolve to filter=safetensors on the hf side -
        that query fires ONCE, not twice (a result's badge comes from its own
        metadata, so the query's type is irrelevant to correctness)."""
        urls = _urls_capture(monkeypatch, [{"id": "org/x", "downloads": 1,
                                            "tags": ["safetensors"]}])
        hf_search("x", limit=5, formats=["hf"], model_types=["vae", "text-encoder"])
        assert len(urls) == 1
        assert "filter=safetensors" in urls[0]

    def test_multi_type_merges_and_dedupes_by_repo_id(self, monkeypatch):
        """llm + embedding run distinct queries; a repo surfacing in both appears
        once, interleaved, badged from its own metadata."""
        import json as _json
        import urllib.parse

        def fake(url, **kw):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            if q.get("filter") == "transformers":
                payload = [{"id": "shared/repo", "downloads": 9,
                            "pipeline_tag": "text-generation", "tags": ["transformers"]},
                           {"id": "llm/only", "downloads": 8, "tags": ["transformers"]}]
            else:   # embedding -> feature-extraction
                payload = [{"id": "shared/repo", "downloads": 9,
                            "pipeline_tag": "feature-extraction", "tags": ["sentence-transformers"]},
                           {"id": "emb/only", "downloads": 7, "tags": ["sentence-transformers"]}]
            return url, "application/json", _json.dumps(payload).encode("utf-8")

        monkeypatch.setattr("localm.netpolicy.safe_fetch_bytes", fake)
        results = hf_search("x", limit=10, formats=["hf"], model_types=["llm", "embedding"])
        ids = [r["id"] for r in results]
        assert ids.count("shared/repo") == 1          # de-duped across the two queries
        assert {"shared/repo", "llm/only", "emb/only"} == set(ids)

    def test_empty_valid_types_raises(self, monkeypatch):
        _mock_fetch(monkeypatch, [])
        with pytest.raises(DiscoverError, match="model type"):
            hf_search("x", model_types=["bogus", "also-bad"])

    def test_unknown_singular_model_type_raises(self, monkeypatch):
        _mock_fetch(monkeypatch, [])
        with pytest.raises(DiscoverError, match="model type"):
            hf_search("x", model_type="bogus")


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
        # Enumeration fields only, NOT dict equality: entries also carry
        # "free_scope", whose value depends on the real host. TestFreeScope below
        # covers that field directly, with the source stubbed.
        assert [{k: g[k] for k in ("index", "name", "total", "free")} for g in gpus] == [
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
        # Both the isolated-torch probe and the nvidia-smi fallback now spawn
        # via subprocess.Popen. The isolated-torch call sees this same CSV
        # text, which is not valid JSON: it correctly reports "unusable
        # reply" (None) and falls through to nvidia-smi, whose call gets the
        # identical mocked reply and parses it as CSV - matching the
        # fallback chain this test exercises.
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.return_value = (csv, "")
        fake_popen.return_value.returncode = 0
        with patch.dict(sys.modules, {"torch": None}), \
             patch("subprocess.Popen", fake_popen):
            gpus = list_gpus()
        assert len(gpus) == 2
        # free_scope "device" is part of this path's contract: nvidia-smi's
        # memory.free is the whole board's across every process, so it needs no
        # correction on any platform.
        assert gpus[0] == {"index": 0, "name": "NVIDIA RTX 4090",
                            "total": 24576 * 1024 ** 2, "free": 20000 * 1024 ** 2,
                            "free_scope": "device"}
        assert gpus[1]["index"] == 1
        assert gpus[1]["name"] == "NVIDIA RTX 3060"

    def test_empty_when_nothing_available(self):
        with patch.dict(sys.modules, {"torch": None}), \
             patch("subprocess.Popen", side_effect=FileNotFoundError):
            assert list_gpus() == []

    def test_empty_when_torch_sees_no_cuda(self):
        fake = MagicMock()
        fake.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake}), \
             patch("subprocess.Popen", side_effect=FileNotFoundError):
            assert list_gpus() == []

    def test_nvidia_smi_wedged_post_kill_drain_is_bounded(self, monkeypatch):
        """Same defect as _torch_gpus_isolated's own drain: subprocess.run's
        Windows kill-path drains the pipes with a SECOND communicate() call
        carrying no timeout of its own, which can block forever if nvidia-smi
        (or a process it spawned) leaves the pipe open. That would wedge the
        shared single-flight GPU-probe lock (_gpu_probe_inflight) permanently.
        The drain here must be bounded, not left open-ended."""
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: True)
        import subprocess
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.side_effect = [
            subprocess.TimeoutExpired("nvidia-smi", 5.0),
            subprocess.TimeoutExpired("nvidia-smi", 3.0),
        ]
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        assert discover._list_gpus_probe() == []

        fake_popen.return_value.kill.assert_called_once()
        calls = fake_popen.return_value.communicate.call_args_list
        assert len(calls) == 2
        assert calls[1].kwargs.get("timeout") is not None, (
            "the post-kill drain has no timeout of its own - a wedged "
            "nvidia-smi (or a process it spawned) holding the pipe open can "
            "block this forever")


class TestListGpusSafety:
    """The safe-by-construction guarantee: list_gpus() must never block its
    caller for longer than its deadline even when the GPU driver probe wedges,
    while ALWAYS returning a fresh reading (no stale free-VRAM)."""

    def test_deadline_bounds_a_wedged_probe(self, monkeypatch):
        import threading
        import time

        release = threading.Event()

        def _wedged():
            release.wait(10)   # simulate a stuck native driver call
            return [{"index": 0, "name": "X", "total": 1, "free": 1}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _wedged)
        t0 = time.monotonic()
        # No prior good value -> [] fallback, returned within the deadline.
        result = list_gpus(deadline=0.3)
        elapsed = time.monotonic() - t0
        release.set()
        assert elapsed < 2.0, f"list_gpus blocked {elapsed:.1f}s past its deadline"
        assert result == []      # no known value yet -> safe "unknown"

    def test_every_call_reprobes_no_stale_free(self, monkeypatch):
        """No freshness cache: successive calls must reflect the LIVE reading,
        never a stale one. switch_engine's eviction loop / wait_for_vram_release
        polls free-VRAM to confirm a native free landed, and a stale value there
        would defeat the over-eviction guard."""
        seq = [
            [{"index": 0, "name": "A", "total": 8, "free": 2}],   # tight
            [{"index": 0, "name": "A", "total": 8, "free": 7}],   # freed up
        ]
        calls = {"n": 0}

        def _probe():
            i = min(calls["n"], len(seq) - 1)
            calls["n"] += 1
            return list(seq[i])

        monkeypatch.setattr("localm.discover._list_gpus_probe", _probe)
        first = list_gpus()
        second = list_gpus()
        assert first[0]["free"] == 2
        assert second[0]["free"] == 7, "second call returned a stale (cached) free"
        assert calls["n"] == 2, "a call was served from cache instead of re-probing"

    def test_thread_start_failure_degrades_and_resets_guard(self, monkeypatch):
        """If the probe thread cannot be spawned (OS thread exhaustion), the call
        must NOT propagate a 500 and must NOT leave the in-flight guard stuck True
        (which would freeze GPU detection for the process lifetime with no
        self-heal). It degrades to last-known-good and re-arms for a later retry."""
        from localm import discover

        good = [{"index": 0, "name": "A", "total": 8, "free": 8}]
        monkeypatch.setattr("localm.discover._list_gpus_probe", lambda: list(good))
        assert list_gpus() == good      # record last-known-good

        class _BoomThread:
            def __init__(self, *a, **k):
                pass

            def start(self):
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr("localm.discover.threading.Thread", _BoomThread)
        result = list_gpus()            # must not raise
        assert result == good           # degraded to last-known-good
        assert discover._gpu_probe_inflight is False   # re-armed for a later retry

    def test_serves_last_known_good_when_probe_wedges(self, monkeypatch):
        import threading
        import time

        good = [{"index": 0, "name": "A", "total": 8, "free": 8}]
        monkeypatch.setattr("localm.discover._list_gpus_probe", lambda: list(good))
        assert list_gpus() == good     # a successful probe records last-known-good

        release = threading.Event()

        def _wedged():
            release.wait(10)
            return [{"index": 9, "name": "late", "total": 1, "free": 1}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _wedged)
        t0 = time.monotonic()
        fallback = list_gpus(deadline=0.3)   # this probe wedges
        elapsed = time.monotonic() - t0
        release.set()
        assert elapsed < 2.0
        assert fallback == good, "a wedged probe must serve the last-known-good value, not []"

    def test_reset_orphans_an_abandoned_probe_so_it_cannot_bleed(self, monkeypatch):
        """An abandoned probe must not land its reading after a reset has retired it.

        A probe that overruns its deadline is ABANDONED, not cancelled (a wedged
        native call cannot be interrupted from Python), so that thread outlives the
        reset and can write _gpu_last_good afterwards, landing inside whatever
        ran next: a cold ROCm init (~6.5s) overruns a 4s deadline, so a real
        card arrives inside later tests that assert a fake or empty reading.

        Guards the epoch fence in _reset_gpu_probe_cache. Clearing the globals
        alone cannot fix this, since the write happens after the clear."""
        import threading
        import time

        from localm import discover

        landed = threading.Event()

        def _slow_probe():
            time.sleep(0.4)          # still running when the reset below lands
            landed.set()
            return [{"index": 0, "name": "STRAGGLER", "total": 1, "free": 1}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow_probe)
        assert list_gpus(deadline=0.05) == []      # overruns -> thread abandoned

        # The next test's autouse fixture, while the probe thread is still running.
        discover._reset_gpu_probe_cache()

        assert landed.wait(5), "probe thread never finished; test cannot conclude"
        time.sleep(0.2)   # give the straggler's write every chance to land
        with discover._gpu_probe_lock:
            leaked = discover._gpu_last_good
        assert leaked is None, (
            f"an abandoned probe wrote {leaked!r} into _gpu_last_good AFTER the "
            f"reset retired it - it would be served to the next test/caller as a "
            f"last-known-good reading")


class TestListGpusTimeoutStatus:
    """A timed-out cold probe must be DISTINGUISHABLE from a genuine empty
    result so a blocking caller can retry / report accurately, instead of
    misattributing a slow driver init to 'no GPU / no torch'. The FIRST cold
    torch.cuda/HIP call initializes the ROCm/CUDA driver (~6.5s) and can overrun
    a short deadline, so list_gpus() returns [] just like a no-torch box."""

    def test_short_deadline_times_out_to_empty_with_timeout_status(self, monkeypatch):
        import threading
        import time

        release = threading.Event()

        def _slow():
            release.wait(10)     # a cold driver init that overruns the short deadline
            return [{"index": 0, "name": "GPU0", "total": 8, "free": 8}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow)
        t0 = time.monotonic()
        gpus, status = list_gpus(deadline=0.2, return_status=True)
        elapsed = time.monotonic() - t0
        release.set()            # let the abandoned probe thread finish now
        assert elapsed < 2.0, f"list_gpus blocked {elapsed:.1f}s past its deadline"
        assert gpus == []
        assert status == GPU_PROBE_TIMEOUT

    def test_generous_deadline_lets_a_slow_cold_probe_complete(self, monkeypatch):
        import time

        def _slow():
            time.sleep(0.4)      # a cold init that beats a generous (CLI) deadline
            return [{"index": 0, "name": "GPU0", "total": 8, "free": 8}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow)
        gpus, status = list_gpus(deadline=3.0, return_status=True)
        assert status == GPU_PROBE_OK
        assert gpus == [{"index": 0, "name": "GPU0", "total": 8, "free": 8}]

    def test_completed_empty_probe_is_ok_not_timeout(self, monkeypatch):
        """A probe that COMPLETES and finds nothing is authoritative 'no GPU'
        (status OK) - the real no-torch/no-nvidia-smi box - and must NOT be
        conflated with a timeout."""
        monkeypatch.setattr("localm.discover._list_gpus_probe", lambda: [])
        gpus, status = list_gpus(deadline=3.0, return_status=True)
        assert gpus == []
        assert status == GPU_PROBE_OK

    def test_return_status_false_keeps_plain_list_contract(self, monkeypatch):
        """The default call returns a bare list (every existing caller and the
        ~28 test files that patch list_gpus rely on it), never a tuple."""
        good = [{"index": 0, "name": "A", "total": 8, "free": 8}]
        monkeypatch.setattr("localm.discover._list_gpus_probe", lambda: list(good))
        assert list_gpus() == good
        assert list_gpus(deadline=3.0) == good

    def test_default_deadline_tolerates_a_cold_driver_init(self):
        """The DEFAULT deadline sits ABOVE a legitimate cold ROCm/CUDA driver
        init, with real margin, so a bare list_gpus() on a cold driver returns
        its real reading instead of timing out into [] / last-known-good."""
        assert _GPU_PROBE_DEADLINE >= 10.0
        # The blocking-caller name stays UNIFIED with the default.
        assert _GPU_PROBE_CLI_DEADLINE == _GPU_PROBE_DEADLINE

    def test_default_deadline_lets_a_cold_init_length_probe_complete(self):
        """Behavioral half of the constant guard above: a probe that takes longer
        than a 4.0s cap (a realistic cold driver init) must COMPLETE at the
        default deadline and hand back its real reading, not time out into the
        []/"no GPU" misreport."""
        import time
        from localm import discover

        good = [{"index": 0, "name": "COLD", "total": 8, "free": 8}]

        def _cold_init():
            time.sleep(4.4)   # just over the retired 4.0s cap
            return list(good)

        discover._reset_gpu_probe_cache()   # own the in-flight slot for sure
        orig = discover._list_gpus_probe
        discover._list_gpus_probe = _cold_init
        try:
            gpus, status = list_gpus(return_status=True)   # DEFAULT deadline
        finally:
            discover._list_gpus_probe = orig
            discover._reset_gpu_probe_cache()
        assert status == GPU_PROBE_OK, (
            f"a cold-init-length probe must complete at the default deadline, "
            f"got {status!r}")
        assert gpus == good

    def test_vram_capacity_forwards_deadline_so_a_cold_probe_completes(self, monkeypatch):
        """A pre-load probe running at a 4s cap lets a cold driver init (~4.6s,
        up to ~6.5s per discover.py's own comment) TIME OUT and serve no reading,
        which makes switch_engine's gate treat the box as unmeasurable and skip
        the VRAM check.

        This pins the mechanism end to end through the plumbing: the SAME slow
        probe TIMES OUT at the default cap (no 'free') but COMPLETES with a real
        reading when vram_capacity is given the generous deadline. If
        vram_capacity stopped forwarding the deadline, the second arm would time
        out too and this goes red."""
        from localm import discover
        import time

        def _slow_cold_probe():
            time.sleep(1.0)   # overruns a short cap, beats a generous one
            return [{"index": 0, "name": "GPU0", "total": 16, "free": 9}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow_cold_probe)
        monkeypatch.setattr("localm.config.load_config", lambda: {})  # no split

        # ARM A: default cap (0.2s here) -> times out -> no 'free' -> gate would skip.
        discover._reset_gpu_probe_cache()
        cap_a, status_a = discover.vram_capacity(return_status=True, deadline=0.2)
        assert status_a == GPU_PROBE_TIMEOUT
        assert cap_a.get("free") is None, (
            "a timed-out probe must not present a 'free' figure it did not measure")

        # ARM B: generous deadline -> the same probe completes -> real reading.
        discover._reset_gpu_probe_cache()
        cap_b, status_b = discover.vram_capacity(return_status=True, deadline=3.0)
        assert status_b == GPU_PROBE_OK
        assert cap_b.get("free") == 9, (
            "the generous deadline must thread through vram_capacity so the cold "
            f"probe completes and the gate gets a real reading; got {cap_b}")

    def test_deadline_forwarded_when_split_degrades_to_single_device(self, monkeypatch):
        """vram_capacity's docstring promises the deadline is forwarded on EVERY
        path. The split-configured-but-degraded-to-<2-devices fallback (a device
        vanished / was never present) dropped it and re-probed at the default 4s cap,
        so a cold init on that specific config could still time out.

        Pinned with a TRACER, not with timings. A timing version does NOT
        guard: this path probes TWICE (vram_capacity's own split probe, then
        vram_info's inside the fallback), and any probe short enough to keep the test
        fast also completes under the 4s DEFAULT cap - so dropping the forwarding
        would leave it green. Worse, the first probe's abandoned thread holds the
        in-flight slot, so the second returns BUSY rather than TIMEOUT and a
        status-based assertion pins the wrong thing. The tracer asserts what actually
        matters: BOTH probes on this path receive the caller's deadline.

        Mutation: drop `deadline` from _vi() (or from _list_gpus_kw) -> the second
        recorded deadline is no longer the caller's -> RED."""
        from localm import discover
        seen = []

        def _tracer(*, deadline=discover._GPU_PROBE_DEADLINE, return_status=False,
                    wait_for_inflight=False):
            seen.append(deadline)
            # Only device 0 exists, so the configured [0, 5] split resolves to <2
            # devices -> vram_capacity takes the degrade fallback under test.
            gpus = [{"index": 0, "name": "GPU0", "total": 16, "free": 9}]
            return (gpus, GPU_PROBE_OK) if return_status else gpus

        monkeypatch.setattr("localm.discover.list_gpus", _tracer)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 5]})

        # SENTINEL deadline, not _GPU_PROBE_CLI_DEADLINE: the CLI constant equals
        # the tracer's default, so asserting the constant is tautological (a
        # dropped forwarding records the default, which compares equal). Only a
        # value nothing defaults to makes the forwarding observable.
        cap, status = discover.vram_capacity(return_status=True, deadline=7.7)

        assert status == GPU_PROBE_OK and cap.get("free") == 9, (
            f"degrade fallback should still return the single device's reading; "
            f"got {status} {cap}")
        assert len(seen) == 2, (
            f"the degrade path probes twice (split probe, then vram_info's); got "
            f"{len(seen)} - the path under test may not be reached")
        assert seen == [7.7] * 2, (
            "BOTH probes on the degrade path must get the caller's deadline; the "
            f"fallback dropping it is the bug this pins. got {seen}")

    def test_vram_capacity_forwards_wait_for_inflight(self, monkeypatch):
        """switch_engine's gate passes wait_for_inflight=True so a load that
        races a concurrent probe (the GUI stats heartbeat holding the slot
        through a cold init) JOINS it for a real reading instead of taking an
        instant BUSY and refusing spuriously. That only works if vram_capacity
        FORWARDS the flag to list_gpus. This pins the forwarding with a tracer;
        the join mechanism itself is tested elsewhere. Dropping
        wait_for_inflight from _vi() / _list_gpus_kw turns the positive assert
        red."""
        from localm import discover
        seen = {}

        def _tracer(*, deadline=discover._GPU_PROBE_DEADLINE, return_status=False,
                    wait_for_inflight=False):
            seen["wait_for_inflight"] = wait_for_inflight
            seen["deadline"] = deadline
            gpus = [{"index": 0, "name": "A", "total": 16, "free": 9}]
            return (gpus, GPU_PROBE_OK) if return_status else gpus

        monkeypatch.setattr("localm.discover.list_gpus", _tracer)
        monkeypatch.setattr("localm.config.load_config", lambda: {})  # no split

        # Sentinel deadline for the same reason as the degrade test above: the
        # unified constant equals the tracer's default, so only a value nothing
        # defaults to proves the forwarding happened.
        discover.vram_capacity(return_status=True, deadline=7.7,
                               wait_for_inflight=True)
        assert seen["wait_for_inflight"] is True, (
            "vram_capacity must forward wait_for_inflight to list_gpus, or the gate's "
            "join is a no-op and the GUI cold-first-load still refuses spuriously")
        assert seen["deadline"] == 7.7

        # Negative: the default (every non-gate caller) must NOT join - joining is
        # an opt-in for off-loop callers only.
        seen.clear()
        discover.vram_capacity(return_status=True)
        assert seen.get("wait_for_inflight") is False


class TestListGpusInconclusiveStatus:
    """Once the isolated-torch latch has engaged AND nvidia-smi also cannot
    answer - the realistic case is an AMD or Intel card whose torch wedges - a
    probe completing with [] and status GPU_PROBE_OK is indistinguishable from a
    genuine no-GPU box. status must be GPU_PROBE_INCONCLUSIVE instead, and ONLY
    in that specific combination."""

    def _blind_nvidia_smi(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(subprocess, "Popen",
                            MagicMock(side_effect=FileNotFoundError("no nvidia-smi")))

    def test_latch_engaged_and_nvidia_smi_blind_reports_inconclusive(self, monkeypatch):
        """The scenario the gap named: torch wedges THIS round (engaging the
        latch as a side effect) and nvidia-smi (NVIDIA-only) cannot see the
        AMD/Intel card either."""
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
        monkeypatch.setattr(discover, "_torch_gpus_isolated",
                            MagicMock(side_effect=discover._IsolatedTorchWedged))
        self._blind_nvidia_smi(monkeypatch)

        gpus, status = list_gpus(return_status=True)

        assert gpus == []
        assert status == discover.GPU_PROBE_INCONCLUSIVE, (
            "torch could not be asked (latched) and nvidia-smi found nothing - "
            "an AMD/Intel box's real answer is unknown, not 'no GPU'")
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unavailable is True   # the latch engaged

    def test_fires_control_genuine_no_gpu_box_stays_ok(self, monkeypatch):
        """FIRES-CONTROL: a real, honest empty answer from torch (no latch) plus
        a blind nvidia-smi is a GENUINE no-GPU box and must NOT be downgraded to
        inconclusive - a change that made every empty reading inconclusive would
        still pass the test above but must fail this one."""
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
        monkeypatch.setattr(discover, "_torch_gpus_isolated", lambda: [])
        self._blind_nvidia_smi(monkeypatch)

        gpus, status = list_gpus(return_status=True)

        assert gpus == []
        assert status == discover.GPU_PROBE_OK, (
            "torch conclusively answered 'no device' - this must stay a genuine "
            "no-GPU reading, not be swept into inconclusive")
        with discover._gpu_probe_lock:
            assert discover._isolated_torch_unavailable is False

    def test_preexisting_latch_with_a_conclusive_nvidia_smi_stays_ok(self, monkeypatch):
        """The actual case the isolation was built for (sm_120): torch is ALREADY
        latched-unavailable from an earlier probe, but nvidia-smi still finds the
        real hardware this round. A non-empty reading is conclusive regardless of
        the latch - it must never be downgraded just because torch sat out."""
        monkeypatch.setattr(discover, "_isolated_torch_unavailable", True)
        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed", lambda: False)
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: False)
        monkeypatch.setattr(
            discover, "_torch_gpus_isolated",
            lambda: pytest.fail("latch was engaged; must not respawn the child"))
        import subprocess
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.return_value = (
            "0, RTX 4090, 24576, 20000\n", "")
        fake_popen.return_value.returncode = 0
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        gpus, status = list_gpus(return_status=True)

        assert status == discover.GPU_PROBE_OK
        assert gpus and gpus[0]["name"] == "RTX 4090"


class TestTorchGpusResidentBounded:
    """`sys.modules` gains a module's entry BEFORE that module's body finishes
    running, so `_torch_is_resident()` can read True while another thread is
    still mid-import of torch. `_torch_gpus_resident_bounded` must not let a
    caller block on that other thread's import lock for its full duration."""

    def test_returns_the_real_value_when_the_read_is_fast(self, monkeypatch):
        monkeypatch.setattr(
            discover, "_torch_gpus_resident",
            lambda: [{"index": 0, "name": "FAST", "total": 1, "free": 1}])

        result = discover._torch_gpus_resident_bounded(timeout=2.0)

        assert result == [{"index": 0, "name": "FAST", "total": 1, "free": 1}]

    def test_returns_empty_without_blocking_past_the_timeout(self, monkeypatch):
        import time

        def _wedged_on_the_import_lock():
            time.sleep(5.0)
            return [{"index": 0, "name": "TOO-LATE", "total": 1, "free": 1}]

        monkeypatch.setattr(discover, "_torch_gpus_resident",
                            _wedged_on_the_import_lock)

        t0 = time.monotonic()
        result = discover._torch_gpus_resident_bounded(timeout=0.1)
        elapsed = time.monotonic() - t0

        assert result == []
        assert elapsed < 1.0, (
            f"took {elapsed:.2f}s against a 0.1s timeout - blocked on the "
            f"other thread's read instead of returning at the bound")

    def test_list_gpus_probe_falls_through_to_nvidia_smi_on_overrun(
            self, monkeypatch):
        """Integration point: _list_gpus_probe must not inherit an unbounded
        block just because _torch_is_resident() said True."""
        import functools
        import subprocess
        import time

        monkeypatch.setattr(discover, "_torch_gpu_probe_known_doomed",
                            lambda: False)
        monkeypatch.setattr(discover, "_torch_is_resident", lambda: True)
        monkeypatch.setattr(
            discover, "_torch_gpus_resident_bounded",
            functools.partial(discover._torch_gpus_resident_bounded, timeout=0.1))

        def _slow_import_then_answer():
            time.sleep(3.0)
            return []

        monkeypatch.setattr(discover, "_torch_gpus_resident",
                            _slow_import_then_answer)
        fake_popen = MagicMock()
        fake_popen.return_value.communicate.return_value = (
            "0, FALLBACK-GPU, 8192, 4096\n", "")
        fake_popen.return_value.returncode = 0
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        t0 = time.monotonic()
        out = discover._list_gpus_probe()
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, (
            f"took {elapsed:.2f}s - the resident-read overrun blocked "
            f"_list_gpus_probe instead of falling through")
        assert out and out[0]["name"] == "FALLBACK-GPU", (
            "did not reach the nvidia-smi fallback after the bounded "
            "resident read overran")


class TestListGpusJoinInflight:
    """A patient off-loop caller must be able to JOIN a probe already in flight
    (opt-in wait_for_inflight), not just get an instant BUSY.

    A longer deadline alone is not enough: on a cold box the FIRST probe
    (typically the GUI's /api/stats heartbeat, off-loaded) holds
    _gpu_probe_inflight for the entire ~4.6s cold ROCm/CUDA init. A model-load
    probe arriving in that window hits the in-flight guard and returns BUSY +
    last-known-good in ~0s WITHOUT probing - the identical short-circuit a
    long-deadline RETRY hits. switch_engine's VRAM-fit gate then sees free=None
    -> measurable=False -> loads unchecked even with a long deadline, because it
    never got to run its own probe. Joining the in-flight probe is what lets the
    long deadline pay off in the concurrent case. The mutation test below (flag
    OFF -> still BUSY) keeps these from being tautological: the join, not the
    deadline, is what changes the outcome."""

    _GOOD = [{"index": 0, "name": "GPU0", "total": 8, "free": 4}]

    @pytest.fixture(autouse=True)
    def _reset_probe_state(self):
        """These tests leave an abandoned probe thread in flight (the whole
        point: a caller that overran its deadline). Reset before and
        after each so a straggler's late write to _gpu_last_good cannot bleed into
        a sibling test - the epoch fence in _reset_gpu_probe_cache is what makes
        that late write a no-op."""
        from localm import discover
        discover._reset_gpu_probe_cache()
        yield
        discover._reset_gpu_probe_cache()

    def test_patient_caller_joins_and_gets_a_real_reading(self, monkeypatch):
        import threading
        import time

        started = threading.Event()
        release = threading.Event()

        def _slow():
            started.set()
            release.wait(10)          # a cold init still running when the joiner arrives
            return list(self._GOOD)

        # Kick off the in-flight probe via a starter that times out at 0.15s.
        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow)
        starter = {}

        def _starter():
            starter["res"] = list_gpus(deadline=0.15, return_status=True)

        st = threading.Thread(target=_starter)
        st.start()
        assert started.wait(2), "probe thread never started"
        st.join()
        assert starter["res"][1] == GPU_PROBE_TIMEOUT

        # Joiner: patient, opts in -> joins the still-running probe.
        joiner = {}

        def _joiner():
            t0 = time.monotonic()
            g, s = list_gpus(deadline=5.0, return_status=True,
                             wait_for_inflight=True)
            joiner["res"] = (g, s, time.monotonic() - t0)

        jt = threading.Thread(target=_joiner)
        jt.start()
        time.sleep(0.2)               # joiner is now blocked on the in-flight probe
        release.set()                 # probe completes -> joiner must wake with it
        jt.join()
        gpus, status, elapsed = joiner["res"]
        assert status == GPU_PROBE_OK, "a joined, completed probe is a fresh OK reading"
        assert gpus == self._GOOD, "the joiner must get the probe's REAL value"
        assert elapsed < 4.0, (
            f"joiner waited {elapsed:.1f}s - it should have woken when the probe "
            f"landed, not waited out its 5s deadline")

    def test_without_opt_in_a_concurrent_caller_still_gets_busy(self, monkeypatch):
        """MUTATION guard: the SAME concurrent scenario, but the second caller does
        NOT opt in -> it must still short-circuit on BUSY in ~0s. This is the bug
        the join fixes; if this ever returns OK, wait_for_inflight has been made
        the unconditional default and the event-loop protection is gone."""
        import threading
        import time

        started = threading.Event()
        release = threading.Event()

        def _slow():
            started.set()
            release.wait(10)
            return list(self._GOOD)

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow)

        def _starter():
            list_gpus(deadline=0.15, return_status=True)

        st = threading.Thread(target=_starter)
        st.start()
        assert started.wait(2)
        st.join()

        t0 = time.monotonic()
        gpus, status = list_gpus(deadline=5.0, return_status=True)  # no opt-in
        elapsed = time.monotonic() - t0
        release.set()
        assert status == GPU_PROBE_BUSY, (
            "a concurrent caller that did NOT opt in must still get BUSY")
        assert elapsed < 1.0, (
            f"BUSY must be instant ({elapsed:.2f}s) - it never waited on the probe")

    def test_joiner_on_a_permanent_wedge_times_out_at_its_own_deadline(self, monkeypatch):
        """A join must be BOUNDED by the joiner's own deadline and must never spawn
        a second probe: on a truly wedged driver the joiner returns TIMEOUT at its
        deadline (not a hang, not a pile-on)."""
        import threading
        import time

        started = threading.Event()
        release = threading.Event()   # never set within the test -> permanent wedge

        def _wedged():
            started.set()
            release.wait(30)
            return list(self._GOOD)

        monkeypatch.setattr("localm.discover._list_gpus_probe", _wedged)

        def _starter():
            list_gpus(deadline=0.15, return_status=True)

        st = threading.Thread(target=_starter)
        st.start()
        assert started.wait(2)
        st.join()

        t0 = time.monotonic()
        gpus, status = list_gpus(deadline=0.5, return_status=True,
                                 wait_for_inflight=True)
        elapsed = time.monotonic() - t0
        release.set()                 # let the abandoned probe finish now
        assert status == GPU_PROBE_TIMEOUT
        assert 0.4 < elapsed < 2.0, (
            f"joiner should time out at ~0.5s, not hang or return early ({elapsed:.2f}s)")

    def test_wait_for_inflight_with_no_probe_running_starts_its_own(self, monkeypatch):
        """When nothing is in flight, wait_for_inflight is a no-op: the caller just
        runs its own probe (the headless/first-caller path), so the flag is safe to
        pass unconditionally from an off-loop caller."""
        import time

        def _slow():
            time.sleep(0.3)
            return list(self._GOOD)

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow)
        gpus, status = list_gpus(deadline=3.0, return_status=True,
                                 wait_for_inflight=True)
        assert status == GPU_PROBE_OK
        assert gpus == self._GOOD

    def test_spawn_failure_wakes_a_joiner_instead_of_hanging(self, monkeypatch):
        """If the STARTER publishes its join handles and THEN fails to spawn the
        probe thread, a caller that joined in that window must be woken (BUSY), not
        left waiting out its whole deadline on a probe that will never run. Guards
        the done.set() in the spawn-failure path."""
        import threading
        import time

        real_thread = threading.Thread
        at_spawn = threading.Event()
        let_fail = threading.Event()

        class _BlockingThenFail:
            def __init__(self, *a, **k):
                pass

            def start(self):
                # We are now PAST publishing the join handles, at the spawn point.
                at_spawn.set()
                let_fail.wait(5)
                raise RuntimeError("cannot start new thread (simulated)")

        # Only discover's probe spawn is redirected; the test spawns its own
        # threads via `real_thread` captured above.
        monkeypatch.setattr("localm.discover.threading.Thread", _BlockingThenFail)
        monkeypatch.setattr("localm.discover._list_gpus_probe",
                            lambda: list(self._GOOD))

        starter = {}

        def _starter():
            starter["res"] = list_gpus(deadline=5.0, return_status=True)

        joiner = {}

        def _joiner():
            t0 = time.monotonic()
            g, s = list_gpus(deadline=5.0, return_status=True,
                             wait_for_inflight=True)
            joiner["res"] = (s, time.monotonic() - t0)

        st = real_thread(target=_starter)
        st.start()
        assert at_spawn.wait(2), "starter never reached the spawn point"
        # Starter has published inflight + join handles and is now blocked in
        # start(); a joiner arriving here joins and waits on the starter's `done`.
        jt = real_thread(target=_joiner)
        jt.start()
        time.sleep(0.2)               # joiner is now blocked on the join handle
        let_fail.set()                # starter's spawn now raises
        st.join()
        jt.join()
        assert starter["res"][1] == GPU_PROBE_BUSY
        joiner_status, joiner_elapsed = joiner["res"]
        assert joiner_status == GPU_PROBE_BUSY, (
            "a spawn failure must wake the joiner as BUSY, not leave it hanging")
        assert joiner_elapsed < 3.0, (
            f"joiner hung {joiner_elapsed:.1f}s waiting on a probe that never ran")


@pytest.fixture
def _non_vulkan_host(monkeypatch):
    """Pin the active native backend to NON-vulkan.

    resolve_main_gpu_index/resolve_gpu_split only cross-check a configured index
    against the detected device list when the active backend is NOT vulkan (on
    vulkan, list_gpus() is structurally blind to the real device list, so the
    index is trusted instead - see _native_backend_has_vulkan). That check reads
    the REAL provisioned runtime dir off disk, so without this pin the classes
    below assert the drop-the-unknown-index behaviour on a host that skips it:
    green on an unprovisioned/HIP box, RED on a vulkan-provisioned one (the
    RECOMMENDED universal build), for the same source.

    The vulkan side of the branch is covered by
    TestVulkanBackendIndexPassthrough, and the detector itself by
    TestNativeBackendHasVulkan, so pinning here narrows these classes to the
    branch they test.
    """
    monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: False)


@pytest.mark.usefixtures("_non_vulkan_host")
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

    def test_index_above_sanity_ceiling_falls_back_to_zero_with_warning(
            self, monkeypatch, caplog):
        # Mirrors resolve_gpu_split's ceiling check: an absurd index is rejected
        # even when gpus is unmeasurable (list_gpus() -> []), before it can reach
        # ctypes.c_int32 main_gpu.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [])
        with caplog.at_level("WARNING", logger="localm"):
            idx = resolve_main_gpu_index(500_000)
        assert idx == 0
        assert any("main_gpu_index" in r.message and "ceiling" in r.message
                   for r in caplog.records)

    def test_index_at_ceiling_boundary_is_used(self):
        gpus = [{"index": 0}, {"index": _MAX_GPU_SPLIT_INDEX}]
        assert resolve_main_gpu_index(_MAX_GPU_SPLIT_INDEX, gpus=gpus) == \
            _MAX_GPU_SPLIT_INDEX


@pytest.mark.usefixtures("_non_vulkan_host")
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


@pytest.mark.usefixtures("_non_vulkan_host")
class TestResolveGpuSplit:
    _GPUS = [{"index": 0}, {"index": 1}, {"index": 2}]

    def test_none_configured_returns_empty(self):
        assert resolve_gpu_split(None) == []

    def test_empty_configured_returns_empty(self):
        assert resolve_gpu_split([]) == []

    def test_two_valid_indices_no_ratios_equal_split(self):
        assert resolve_gpu_split([0, 1], gpus=self._GPUS) == [(0, 1.0), (1, 1.0)]

    def test_two_valid_indices_with_matching_ratios_paired_by_index(self):
        assert resolve_gpu_split([0, 1], [3.0, 1.0], gpus=self._GPUS) == \
            [(0, 3.0), (1, 1.0)]

    def test_ratio_still_lines_up_with_its_index_after_a_drop(self):
        # index 5 is unknown and gets dropped; its ratio (9.0, sitting between
        # the other two in the configured lists) must not bleed onto index 0
        # or index 1 - each surviving index keeps ITS OWN configured ratio.
        result = resolve_gpu_split([0, 5, 1], [3.0, 9.0, 1.0], gpus=self._GPUS)
        assert result == [(0, 3.0), (1, 1.0)]

    def test_unknown_index_dropped_with_warning_remaining_valid_still_work(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            result = resolve_gpu_split([0, 1, 7], gpus=self._GPUS[:2])
        assert result == [(0, 1.0), (1, 1.0)]
        assert any("gpu_split_indices" in r.message for r in caplog.records)

    def test_ratio_length_mismatch_falls_back_to_equal_split_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            result = resolve_gpu_split([0, 1], [1.0, 2.0, 3.0], gpus=self._GPUS[:2])
        assert result == [(0, 1.0), (1, 1.0)]
        assert any("gpu_split_ratios" in r.message for r in caplog.records)

    def test_single_valid_survivor_collapses_to_no_split(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            result = resolve_gpu_split([0, 9], gpus=self._GPUS[:1])
        assert result == []
        assert any("gpu_split_indices" in r.message for r in caplog.records)

    def test_duplicate_indices_deduped_first_occurrence_kept(self):
        # "1" appears twice; the SECOND occurrence is dropped as a duplicate,
        # and the surviving order follows first-appearance order (1, 0, 2).
        result = resolve_gpu_split([1, 0, 1, 2], gpus=self._GPUS)
        assert result == [(1, 1.0), (0, 1.0), (2, 1.0)]

    def test_negative_index_returns_empty_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            result = resolve_gpu_split([-1, 0], gpus=self._GPUS)
        assert result == []
        assert any("gpu_split_indices" in r.message and "negative" in r.message
                   for r in caplog.records)

    def test_non_integer_indices_return_empty_with_warning(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            result = resolve_gpu_split(["not-a-number", 0], gpus=self._GPUS)
        assert result == []
        assert any("gpu_split_indices" in r.message for r in caplog.records)

    def test_index_above_sanity_ceiling_returns_empty_with_warning(self, caplog):
        # An absurd index never reaches apply_gpu_split's ctypes allocation, even
        # when gpus is unmeasurable (list_gpus() -> []), so a configured index like
        # 500000 cannot size a 500,001-element tensor_split array.
        with caplog.at_level("WARNING", logger="localm"):
            result = resolve_gpu_split([0, 500_000], gpus=[])
        assert result == []
        assert any("gpu_split_indices" in r.message and "ceiling" in r.message
                   for r in caplog.records)

    def test_index_at_ceiling_boundary_is_allowed(self):
        gpus = [{"index": 0}, {"index": _MAX_GPU_SPLIT_INDEX}]
        assert resolve_gpu_split([0, _MAX_GPU_SPLIT_INDEX], gpus=gpus) == \
            [(0, 1.0), (_MAX_GPU_SPLIT_INDEX, 1.0)]

    def test_unmeasurable_passes_through_unvalidated(self, monkeypatch):
        # gpus not injected and list_gpus() finds nothing (no torch, no
        # nvidia-smi): the configured indices cannot be cross-checked, so they pass
        # through rather than discarding an explicit user choice (same documented
        # boundary as resolve_main_gpu_index).
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [])
        assert resolve_gpu_split([3, 7]) == [(3, 1.0), (7, 1.0)]


def _ggml_lib_name(stem: str) -> str:
    """A ggml backend library filename matching THIS platform's _ggml_glob(), so
    the detector below is exercised through its real glob rather than a stubbed
    one."""
    if sys.platform == "win32":
        return f"ggml-{stem}.dll"
    if sys.platform == "darwin":
        return f"libggml-{stem}.dylib"
    return f"libggml-{stem}.so"


class TestVulkanBackendIndexPassthrough:
    """The vulkan side of the index-validation branch.

    On the vulkan build, list_gpus() cannot enumerate the real device list at
    all (it only sees torch.cuda / nvidia-smi), so a non-empty but
    vulkan-incomplete list must NOT veto an explicitly configured index, which
    would silently drop a VALID device. These pin the backend to vulkan and
    assert the pass-through contract, so the behaviour is tested on every host
    instead of only on whichever runtime happens to be provisioned.
    """

    @pytest.fixture(autouse=True)
    def _vulkan_host(self, monkeypatch):
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan",
                            lambda: True)

    def test_main_gpu_index_absent_from_detected_list_is_trusted(self, caplog):
        # The exact inversion of TestResolveMainGpuIndex's drop test: index 5 is
        # not in the (vulkan-blind) list, so it is trusted, not swapped for 0.
        with caplog.at_level("WARNING", logger="localm"):
            idx = resolve_main_gpu_index(5, gpus=[{"index": 0}, {"index": 1}])
        assert idx == 5
        assert not any("does not match" in r.message for r in caplog.records)

    def test_gpu_split_unknown_index_is_kept(self):
        assert resolve_gpu_split([0, 9], gpus=[{"index": 0}]) == \
            [(0, 1.0), (9, 1.0)]

    def test_gpu_split_ratios_still_pair_with_their_own_index(self):
        # Nothing is dropped on vulkan, so every configured ratio keeps its index.
        assert resolve_gpu_split([0, 5, 1], [3.0, 9.0, 1.0],
                                 gpus=[{"index": 0}, {"index": 1}]) == \
            [(0, 3.0), (5, 9.0), (1, 1.0)]

    def test_apply_main_gpu_trusts_configured_index(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [{"index": 0}])
        mp = SimpleNamespace(main_gpu=0)
        apply_main_gpu(mp, config={"main_gpu_index": 7})
        assert mp.main_gpu == 7

    def test_sanity_ceiling_still_enforced_on_vulkan(self, caplog):
        # The ceiling is checked BEFORE any device-membership branching, so
        # being unable to validate against a device list must NOT wave an absurd
        # index through to the native loader.
        with caplog.at_level("WARNING", logger="localm"):
            idx = resolve_main_gpu_index(500_000, gpus=[{"index": 0}])
        assert idx == 0
        assert any("ceiling" in r.message for r in caplog.records)

    def test_split_sanity_ceiling_still_enforced_on_vulkan(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            result = resolve_gpu_split([0, 500_000], gpus=[{"index": 0}])
        assert result == []
        assert any("ceiling" in r.message for r in caplog.records)

    def test_negative_index_still_rejected_on_vulkan(self, caplog):
        with caplog.at_level("WARNING", logger="localm"):
            idx = resolve_main_gpu_index(-1, gpus=[{"index": 0}])
        assert idx == 0
        assert any("negative" in r.message for r in caplog.records)


class TestNativeBackendHasVulkan:
    """The backend detector reads the real shipped library set, so the runtime
    dir is pinned at a tmp_path and the assertions are on the file set."""

    @pytest.fixture
    def _runtime_dir(self, tmp_path, monkeypatch):
        # _native_backend_has_vulkan imports these INSIDE the function, so patch
        # them on the SOURCE module, not on localm.discover.
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
            lambda: tmp_path)
        return tmp_path

    def test_true_when_vulkan_backend_is_shipped(self, _runtime_dir):
        (_runtime_dir / _ggml_lib_name("base")).write_bytes(b"")
        (_runtime_dir / _ggml_lib_name("vulkan")).write_bytes(b"")
        assert _native_backend_has_vulkan() is True

    def test_false_when_a_non_vulkan_backend_is_shipped(self, _runtime_dir):
        # e.g. the HIP/ROCm build: present, but not vulkan.
        (_runtime_dir / _ggml_lib_name("base")).write_bytes(b"")
        (_runtime_dir / _ggml_lib_name("hip")).write_bytes(b"")
        assert _native_backend_has_vulkan() is False

    def test_false_when_no_backend_libraries_present(self, _runtime_dir):
        assert _native_backend_has_vulkan() is False

    def test_false_when_runtime_dir_unresolved(self, monkeypatch):
        # No native runtime provisioned at all: not vulkan, and must not raise.
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
            lambda: None)
        assert _native_backend_has_vulkan() is False

    def test_false_and_no_raise_when_the_probe_itself_fails(self, monkeypatch):
        # Detection is best-effort: a broken runtime resolution must degrade to
        # "not vulkan" (the validating branch) rather than break model loading.
        def _boom():
            raise OSError("runtime dir exploded")

        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
            _boom)
        assert _native_backend_has_vulkan() is False


@pytest.mark.usefixtures("_non_vulkan_host")
class TestApplyGpuSplit:
    _GPUS = [{"index": 0}, {"index": 1}, {"index": 2}]

    def test_fewer_than_two_valid_entries_leaves_mp_untouched(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        mp = SimpleNamespace(main_gpu=0, tensor_split="SENTINEL_TS",
                             split_mode="SENTINEL_SM")
        result = apply_gpu_split(
            mp, config={"gpu_split_indices": [0], "gpu_split_ratios": None})
        assert result is None
        assert mp.tensor_split == "SENTINEL_TS"
        assert mp.split_mode == "SENTINEL_SM"

    def test_two_valid_entries_sets_split_mode_and_tensor_split(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS[:2])
        # Force the documented-fallback capacity so the "0.0 elsewhere" check
        # below lands on a real, in-bounds slot regardless of what native
        # runtime (if any) happens to be provisioned on the box running this.
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._api.has_max_devices",
            lambda: False)
        mp = SimpleNamespace(main_gpu=0, tensor_split=None, split_mode=0)
        result = apply_gpu_split(
            mp, config={"gpu_split_indices": [0, 1],
                        "gpu_split_ratios": [3.0, 1.0]})
        assert result is not None
        assert mp.split_mode == 1
        assert mp.split_mode == _LLAMA_SPLIT_MODE_LAYER
        floats = ctypes.cast(mp.tensor_split, ctypes.POINTER(ctypes.c_float))
        assert floats[0] == pytest.approx(3.0)
        assert floats[1] == pytest.approx(1.0)
        assert floats[2] == pytest.approx(0.0)

    def test_main_gpu_corrected_when_not_in_split_set(self, monkeypatch, caplog):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        mp = SimpleNamespace(main_gpu=5, tensor_split=None, split_mode=0)
        with caplog.at_level("WARNING", logger="localm"):
            apply_gpu_split(
                mp, config={"gpu_split_indices": [1, 2], "gpu_split_ratios": None})
        assert mp.main_gpu == 1   # first split index
        assert any("main_gpu_index" in r.message for r in caplog.records)

    def test_main_gpu_already_in_split_left_unchanged_no_warning(self, monkeypatch, caplog):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        mp = SimpleNamespace(main_gpu=1, tensor_split=None, split_mode=0)
        with caplog.at_level("WARNING", logger="localm"):
            apply_gpu_split(
                mp, config={"gpu_split_indices": [1, 2], "gpu_split_ratios": None})
        assert mp.main_gpu == 1
        assert not any("main_gpu_index" in r.message for r in caplog.records)

    def test_reads_load_config_when_config_not_passed(self, monkeypatch):
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": None, "gpu_split_ratios": None})
        mp = SimpleNamespace(main_gpu=0, tensor_split=None, split_mode=0)
        result = apply_gpu_split(mp)
        assert result is None
        assert mp.tensor_split is None
        assert mp.split_mode == 0

    def test_capacity_falls_back_to_documented_constant(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS[:2])
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._api.has_max_devices",
            lambda: False)
        mp = SimpleNamespace(main_gpu=0, tensor_split=None, split_mode=0)
        result = apply_gpu_split(
            mp, config={"gpu_split_indices": [0, 1], "gpu_split_ratios": None})
        assert len(result) == _TENSOR_SPLIT_FALLBACK_CAPACITY

    def test_capacity_grows_for_a_configured_index_at_or_above_fallback(self, monkeypatch):
        high_idx = _TENSOR_SPLIT_FALLBACK_CAPACITY   # exactly at the boundary
        gpus = [{"index": 0}, {"index": high_idx}]
        monkeypatch.setattr("localm.discover.list_gpus", lambda: gpus)
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._api.has_max_devices",
            lambda: False)
        mp = SimpleNamespace(main_gpu=0, tensor_split=None, split_mode=0)
        result = apply_gpu_split(
            mp, config={"gpu_split_indices": [0, high_idx],
                        "gpu_split_ratios": None})
        assert len(result) == high_idx + 1
        floats = ctypes.cast(mp.tensor_split, ctypes.POINTER(ctypes.c_float))
        assert floats[high_idx] == pytest.approx(1.0)


@pytest.mark.usefixtures("_non_vulkan_host")
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


@pytest.mark.usefixtures("_non_vulkan_host")
class TestVramCapacitySplitAware:
    """vram_capacity() must sum total/free across a CONFIGURED multi-GPU split
    (2+ valid resolve_gpu_split() devices) instead of vram_info()'s single main-
    GPU number, so a model too big for one GPU alone but fitting split across N
    configured devices is neither refused nor badged against one device's
    capacity."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000},
        {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000},
        {"index": 2, "name": "C", "total": 8_000_000_000, "free": 6_000_000_000},
    ]

    def test_no_split_configured_falls_back_to_vram_info(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": None})
        assert vram_capacity() == vram_info() == {
            "total": 24_000_000_000, "free": 20_000_000_000}

    def test_two_valid_split_devices_sums_total_and_free(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
        assert vram_capacity() == {
            "total": 24_000_000_000 + 12_000_000_000,
            "free": 20_000_000_000 + 10_000_000_000,
        }

    def test_three_valid_split_devices_sums_all_three(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1, 2]})
        assert vram_capacity() == {
            "total": 24_000_000_000 + 12_000_000_000 + 8_000_000_000,
            "free": 20_000_000_000 + 10_000_000_000 + 6_000_000_000,
        }

    def test_config_param_injection_bypasses_load_config(self, monkeypatch):
        """Matches apply_gpu_split()/apply_main_gpu()'s existing config= convention."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS[:2])
        assert vram_capacity(config={"gpu_split_indices": [0, 1]}) == {
            "total": 24_000_000_000 + 12_000_000_000,
            "free": 20_000_000_000 + 10_000_000_000,
        }

    def test_fewer_than_two_valid_devices_falls_back_to_vram_info(self, monkeypatch):
        """Only one of the two configured indices is currently detected -
        resolve_gpu_split degrades this to "no split" (its own contract) - the
        combined-capacity path must degrade the same way, not sum a single device."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS[:1])
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 5]})
        assert vram_capacity() == vram_info() == {
            "total": 24_000_000_000, "free": 20_000_000_000}

    def test_partially_measurable_free_omits_free_but_keeps_total(self, monkeypatch):
        """vram_info()'s own contract: "free" is present only when measurable.
        A split where one device cannot report free must not silently under-
        count by treating the missing value as 0 - omit "free" entirely, same
        as a single unmeasurable-free device would."""
        gpus = [
            {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000},
            {"index": 1, "name": "B", "total": 12_000_000_000},  # no "free" key
        ]
        monkeypatch.setattr("localm.discover.list_gpus", lambda: gpus)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
        result = vram_capacity()
        assert result == {"total": 24_000_000_000 + 12_000_000_000}
        assert "free" not in result

    def test_unmeasurable_gpus_falls_back_to_vram_info_registry_tier(self, monkeypatch):
        """No list_gpus() data at all (GGUF-only/non-NVIDIA install): cannot sum
        per-device capacity, so this must defer to vram_info()'s own registry-
        fallback tier rather than fabricate a combined number from nothing."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [])
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
        monkeypatch.setattr("localm.discover.vram_info", lambda: {"total": 16_000_000_000})
        assert vram_capacity() == {"total": 16_000_000_000}


class _FakeKey:
    """A winreg key handle usable as a context manager."""
    def __init__(self, values=None, subkeys=()):
        self._values = dict(values or {})
        self._subkeys = list(subkeys)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class _FakeWinreg:
    """Minimal winreg stand-in modelling the display-adapter class key, so the
    registry tier of discover.vram_info can be exercised off Windows."""
    HKEY_LOCAL_MACHINE = object()

    def __init__(self, adapters):
        self._adapters = adapters   # {subkey_name: {value_name: value}}

    def OpenKey(self, root, sub):
        if root is self.HKEY_LOCAL_MACHINE:
            return _FakeKey(subkeys=list(self._adapters))   # the class root
        return _FakeKey(values=self._adapters[sub])          # an adapter subkey

    def EnumKey(self, key, i):
        if i < len(key._subkeys):
            return key._subkeys[i]
        raise OSError("no more subkeys")

    def QueryValueEx(self, key, name):
        if name in key._values:
            return (key._values[name], 1)
        raise OSError("value not present")


class TestVramInfoRegistryTierDeviceGlobalFree:
    """discover.vram_info's Windows registry tier (reached torch-less, when
    list_gpus() is empty) must enrich its total-only reading with a DEVICE-GLOBAL
    free from the ADL/PDH usage source - but ONLY when the probe completed empty,
    never on a timeout, and never when the source cannot map the adapter. On a
    GGUF-only install the registry is the only VRAM source, so without the free
    the meter and fit gates see total-only."""

    _TOTAL = 17_163_091_968

    def _arm(self, monkeypatch, *, status, used, name="AMD Radeon RX 6900 XT"):
        monkeypatch.setattr(sys, "platform", "win32")

        def _fake_list_gpus_kw(**kw):
            # Mirror the real signature: a bare [] unless the caller asked for the
            # status tuple (so return_status=False keeps vram_info's status=None).
            return ([], status) if kw.get("return_status") else []
        monkeypatch.setattr("localm.discover._list_gpus_kw", _fake_list_gpus_kw)
        adapters = {
            "notadigit": {},                                  # skipped (not a digit)
            "0000": {"HardwareInformation.qwMemorySize": self._TOTAL,
                     "DriverDesc": name},
        }
        monkeypatch.setitem(sys.modules, "winreg", _FakeWinreg(adapters))
        seen = {}

        def _dg(gpus):
            seen["gpus"] = gpus
            return {0: used} if used is not None else {}
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes", _dg)
        return seen

    def test_completed_empty_probe_adds_device_global_free(self, monkeypatch):
        seen = self._arm(monkeypatch, status=GPU_PROBE_OK, used=3_500_000_000)
        info, st = discover.vram_info(return_status=True)
        assert st == GPU_PROBE_OK
        assert info["total"] == self._TOTAL
        assert info["free"] == self._TOTAL - 3_500_000_000
        assert info["free_scope"] == discover.FREE_SCOPE_DEVICE
        # the adapter name from DriverDesc must reach the usage source so it can
        # authorise the AMD single-adapter pairing (see gpu_usage._gpu_is_amd).
        assert seen["gpus"][0]["name"] == "AMD Radeon RX 6900 XT"

    def test_timed_out_probe_stays_total_only(self, monkeypatch):
        """A timeout means the box is unmeasurable and the pre-load gate skips the
        VRAM check; surfacing an independent ADL number here would silently turn a
        skipped gate into an enforcing one, so the enrichment must not run."""
        seen = self._arm(monkeypatch, status=GPU_PROBE_TIMEOUT, used=3_500_000_000)
        info, st = discover.vram_info(return_status=True)
        assert st == GPU_PROBE_TIMEOUT
        assert info == {"total": self._TOTAL}
        assert "gpus" not in seen, "the usage source must not be consulted on timeout"

    def test_busy_probe_stays_total_only(self, monkeypatch):
        self._arm(monkeypatch, status=GPU_PROBE_BUSY, used=3_500_000_000)
        info, _ = discover.vram_info(return_status=True)
        assert info == {"total": self._TOTAL}

    def test_inconclusive_probe_stays_total_only(self, monkeypatch):
        """An INCONCLUSIVE reading (isolated torch could not be asked and
        nvidia-smi also found nothing) gets the same conservative treatment as
        TIMEOUT/BUSY: gpu_usage.device_global_used_bytes' ADL/PDH mapping is
        not proven independent of the same torch trouble, so the enrichment
        must not run - stay total-only rather than assert a fresh certainty
        this call did not earn."""
        seen = self._arm(monkeypatch, status=GPU_PROBE_INCONCLUSIVE, used=3_500_000_000)
        info, st = discover.vram_info(return_status=True)
        assert st == GPU_PROBE_INCONCLUSIVE
        assert info == {"total": self._TOTAL}
        assert "gpus" not in seen, "the usage source must not be consulted when inconclusive"

    def test_declining_source_stays_total_only(self, monkeypatch):
        """A non-AMD / multi-adapter box: the usage source returns {} rather than
        guess a pairing, and the reading stays honestly total-only."""
        self._arm(monkeypatch, status=GPU_PROBE_OK, used=None,
                  name="NVIDIA GeForce RTX 4090")
        info, _ = discover.vram_info(return_status=True)
        assert info == {"total": self._TOTAL}

    def test_no_status_requested_still_enriches(self, monkeypatch):
        """A fit-badge caller (return_status=False, status is None) still gets the
        free - it never gates on timeout, so the enrichment applies."""
        self._arm(monkeypatch, status=None, used=3_500_000_000)
        info = discover.vram_info()
        assert info["free"] == self._TOTAL - 3_500_000_000
        assert info["free_scope"] == discover.FREE_SCOPE_DEVICE


class TestVramInfoReturnStatus:
    """vram_info(return_status=True) must propagate list_gpus()'s own probe
    status, so a caller reporting a specific VRAM number as CURRENT FACT (not
    just a fit ceiling) can tell a fresh reading from a timed-out or stale one,
    which /v1/models/unload's vram_before/after_bytes reporting depends on."""

    def test_default_call_keeps_plain_dict_contract(self, monkeypatch):
        monkeypatch.setattr("localm.discover._list_gpus_probe",
                             lambda: [{"index": 0, "name": "A", "total": 8, "free": 4}])
        assert vram_info() == {"total": 8, "free": 4}

    def test_return_status_true_reports_ok_on_a_fresh_probe(self, monkeypatch):
        monkeypatch.setattr("localm.discover._list_gpus_probe",
                             lambda: [{"index": 0, "name": "A", "total": 8, "free": 4}])
        info, status = vram_info(return_status=True)
        assert info == {"total": 8, "free": 4}
        assert status == GPU_PROBE_OK

    def test_return_status_true_reports_timeout_on_a_wedged_probe(self, monkeypatch):
        import threading
        release = threading.Event()

        def _slow():
            release.wait(10)
            return [{"index": 0, "name": "A", "total": 8, "free": 4}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow)
        # Explicit short deadline: what is under test is the STATUS PROPAGATION
        # through vram_info on an overrun, not the default deadline's value (the
        # default is cold-init-tolerant and would out-wait this simulated wedge).
        info, status = vram_info(return_status=True, deadline=0.3)
        release.set()
        assert status == GPU_PROBE_TIMEOUT
        # No last-known-good reading yet in this test, so info is the genuinely
        # empty ({} via the no-torch/no-devices tail) - the point being asserted
        # here is the STATUS, not this incidental value.
        assert isinstance(info, dict)


class TestVramCapacityReturnStatus:
    """vram_capacity(return_status=True) must propagate probe status through
    BOTH the single-GPU short-circuit (delegates to vram_info) and the
    multi-GPU split-summed path, not just one of them."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000},
        {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000},
    ]

    def test_no_split_configured_propagates_status_via_vram_info(self, monkeypatch):
        monkeypatch.setattr("localm.discover._list_gpus_probe", lambda: self._GPUS)
        monkeypatch.setattr("localm.config.load_config",
                             lambda: {"gpu_split_indices": None})
        info, status = vram_capacity(return_status=True)
        assert info == {"total": 24_000_000_000, "free": 20_000_000_000}
        assert status == GPU_PROBE_OK

    @pytest.mark.usefixtures("_non_vulkan_host")
    def test_split_configured_propagates_status(self, monkeypatch):
        monkeypatch.setattr("localm.discover._list_gpus_probe", lambda: self._GPUS)
        monkeypatch.setattr("localm.config.load_config",
                             lambda: {"gpu_split_indices": [0, 1]})
        info, status = vram_capacity(return_status=True)
        assert info == {"total": 36_000_000_000, "free": 30_000_000_000}
        assert status == GPU_PROBE_OK

    def test_default_call_keeps_plain_dict_contract_regardless_of_split(self, monkeypatch):
        """The ~28 test files that patch list_gpus()/vram_capacity() with a
        plain no-kwarg stand-in must never see the new kwarg emitted unless they
        opted in, or they raise TypeError: unexpected keyword argument."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr("localm.config.load_config",
                             lambda: {"gpu_split_indices": [0, 1]})
        assert vram_capacity() == {"total": 36_000_000_000, "free": 30_000_000_000}


@pytest.mark.usefixtures("_non_vulkan_host")
class TestSplitDeviceCount:
    """split_device_count(): the DETECTED split size vram_capacity() uses to decide
    combined-vs-single, so a caption/message can name the VRAM basis honestly. Must
    agree with vram_capacity()'s own combined/fall-back decision on the same inputs."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000},
        {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000},
    ]

    def test_no_split_is_zero_without_probing(self, monkeypatch):
        # The common single-GPU path must not even call list_gpus().
        called = {"n": 0}
        monkeypatch.setattr("localm.discover.list_gpus",
                            lambda: called.__setitem__("n", called["n"] + 1) or self._GPUS)
        assert split_device_count({"gpu_split_indices": None}) == 0
        assert called["n"] == 0, "no split configured -> no hardware probe"

    def test_two_valid_devices_counts_two(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        assert split_device_count({"gpu_split_indices": [0, 1]}) == 2

    def test_stale_index_resolves_below_two(self, monkeypatch):
        # Only device 0 detected: matches vram_capacity()'s fall-back to single.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS[:1])
        n = split_device_count({"gpu_split_indices": [0, 5]})
        assert n < 2
        # And it agrees with vram_capacity(): that also degrades to the single GPU.
        assert vram_capacity(config={"gpu_split_indices": [0, 5]}) == vram_info()

    def test_gguf_only_box_no_gpus_is_below_two(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: [])
        assert split_device_count({"gpu_split_indices": [0, 1]}) < 2


class TestAppliedSplitDeviceCount:
    """applied_split_device_count(): the LOADER-TRUTH count (mirrors
    apply_gpu_split's own gate) vs split_device_count()'s DETECTED/labelling count.
    The two AGREE on a non-vulkan box with a detected device list, and DIVERGE
    exactly where the loader really splits but list_gpus() cannot measure it: a
    GGUF-only box (no torch) and the vulkan build."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000},
        {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000},
    ]

    def _vulkan(self, monkeypatch, on):
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: on)

    def test_no_split_is_zero_without_probing(self, monkeypatch):
        # Mirrors split_device_count's no-probe contract: the common single-GPU
        # path must not touch the hardware at all.
        self._vulkan(monkeypatch, False)
        called = {"n": 0}
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1) or self._GPUS)
        assert applied_split_device_count({"gpu_split_indices": None}) == 0
        assert called["n"] == 0, "no split configured -> no hardware probe"

    def test_non_vulkan_two_valid_devices_counts_two(self, monkeypatch):
        self._vulkan(monkeypatch, False)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: self._GPUS)
        assert applied_split_device_count({"gpu_split_indices": [0, 1]}) == 2

    def test_non_vulkan_stale_index_degrades_below_two(self, monkeypatch):
        # Only device 0 detected: resolve_gpu_split drops the stale 5, the loader
        # would NOT split -> 0, matching apply_gpu_split (single-GPU default) and
        # split_device_count. The non-vulkan degrade reasoning is preserved.
        self._vulkan(monkeypatch, False)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: self._GPUS[:1])
        assert applied_split_device_count({"gpu_split_indices": [0, 5]}) == 0

    @pytest.mark.parametrize("indices", [[0, 1], [0, 5]])
    def test_matches_split_device_count_on_non_vulkan(self, monkeypatch, indices):
        # The invariant: identical to the DETECTED count on a non-vulkan box with a
        # detected device list (split_device_count's re-filter is a proven no-op
        # there). Divergence is a vulkan / unmeasurable-only phenomenon.
        self._vulkan(monkeypatch, False)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: self._GPUS)
        cfg = {"gpu_split_indices": indices}
        assert applied_split_device_count(cfg) == split_device_count(cfg)

    def test_vulkan_split_is_two_even_though_detected_collapses(self, monkeypatch):
        # list_gpus() is Vulkan-blind and reports only device 0, but the loader
        # really tensor_splits across [0, 1]. applied_ says 2 (loader truth) while
        # split_device_count collapses to < 2.
        self._vulkan(monkeypatch, True)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: self._GPUS[:1])
        cfg = {"gpu_split_indices": [0, 1]}
        assert applied_split_device_count(cfg) == 2
        assert split_device_count(cfg) < 2   # the DETECTED count still (correctly) collapses

    def test_vulkan_split_two_when_list_gpus_blind_empty(self, monkeypatch):
        self._vulkan(monkeypatch, True)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: [])
        assert applied_split_device_count({"gpu_split_indices": [0, 1]}) == 2

    def test_vulkan_sanity_ceiling_still_enforced(self, monkeypatch):
        # Passthrough on vulkan does NOT mean "trust any integer": an absurd index
        # is still rejected (resolve_gpu_split's ceiling), so the loader is never
        # handed an index past the end of its device array.
        self._vulkan(monkeypatch, True)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: [])
        assert applied_split_device_count({"gpu_split_indices": [0, 500_000]}) == 0

    @pytest.mark.parametrize("vulkan,gpus,indices", [
        (False, _GPUS, [0, 1]),        # non-vulkan detected: both apply and applied split
        (False, _GPUS[:1], [0, 5]),    # non-vulkan stale: neither splits
        (True, _GPUS[:1], [0, 1]),     # vulkan blind: apply splits, applied agrees
    ])
    def test_agrees_with_apply_gpu_split_gate(self, monkeypatch, vulkan, gpus, indices):
        # Pin applied_ against apply_gpu_split's REAL gate, not a re-derivation:
        # apply_gpu_split returns non-None iff it writes a 2+ device tensor_split,
        # and applied_ >= 2 equals that exactly. On the vulkan row
        # split_device_count would say < 2 and disagree with the loader.
        self._vulkan(monkeypatch, vulkan)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: list(gpus))
        # Pin the tensor_split capacity to the documented fallback so this does not
        # depend on whatever native runtime is provisioned (same guard
        # TestApplyGpuSplit uses).
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._api.has_max_devices", lambda: False)
        cfg = {"gpu_split_indices": indices}
        mp = SimpleNamespace(main_gpu=0, tensor_split=None, split_mode=0)
        applied_two_plus = applied_split_device_count(cfg) >= 2
        assert (apply_gpu_split(mp, config=cfg) is not None) == applied_two_plus


class TestGpuSplitShortfallVulkan:
    """gpu_split_shortfall() honest-unknown on the vulkan build: the configured
    split indices are in ggml-vulkan's index space, not torch's, so a per-device
    free-VRAM check would name the WRONG card. It must SKIP the check (return the
    'nothing to block on' sentinel []) and SURFACE the skip at INFO (discoverable
    in a bug report), never presenting an un-run check as passed. The one guard
    covers both callers (the embedder AND http_server's chat path)."""

    # A MIXED box where the UN-guarded check WOULD flag both devices (tiny free).
    _MIXED = [
        {"index": 0, "total": 16_000_000_000, "free": 1_000_000_000},
        {"index": 1, "total": 16_000_000_000, "free": 1_000_000_000},
    ]

    def test_vulkan_skips_per_device_check_and_logs_info(self, monkeypatch, caplog):
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: True)
        monkeypatch.setattr("localm.discover.list_gpus", lambda *a, **k: self._MIXED)
        cfg = {"gpu_split_indices": [0, 1]}
        with caplog.at_level(logging.INFO, logger="localm"):
            result = gpu_split_shortfall(8_000_000_000, cfg)
        assert result == []
        info = [r for r in caplog.records
                if r.levelno == logging.INFO and "GPU-SPLIT-VKINDEX" in r.getMessage()]
        # The skip is logged at INFO, so it reaches a bug report.
        assert info, "the vulkan skip must be surfaced at INFO (reaches a bug report), not debug/silence"

    def test_vulkan_skip_does_not_probe_torch(self, monkeypatch):
        # The guard sits BEFORE the list_gpus() probe, so a torch-blind vulkan box
        # pays no probe cost for a check it structurally cannot do.
        monkeypatch.setattr("localm.discover._native_backend_has_vulkan", lambda: True)
        called = {"n": 0}
        monkeypatch.setattr(
            "localm.discover.list_gpus",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1) or self._MIXED)
        gpu_split_shortfall(8_000_000_000, {"gpu_split_indices": [0, 1]})
        assert called["n"] == 0


@pytest.mark.usefixtures("_non_vulkan_host")
class TestGpuSplitShortfall:
    """gpu_split_shortfall(): vram_capacity()'s AGGREGATE check alone is not
    enough for a GGUF-backend load - with PINNED gpu_split_ratios,
    apply_gpu_split() divides a model by that static per-config ratio with no
    live per-device capacity awareness, so an asymmetric split (e.g. another
    already-loaded model sits on one device more than another) can pass the
    aggregate check while one device's actual share is short. This is the
    per-device gate that catches that case. With ratios UNSET the gate (and the
    loader) use the auto free-VRAM-proportional shares instead, so the
    static-share cases here pin ratios explicitly."""

    _GPUS = [
        {"index": 0, "name": "A", "total": 16_000_000_000, "free": 2_000_000_000},
        {"index": 1, "name": "B", "total": 16_000_000_000, "free": 14_000_000_000},
    ]

    def test_no_split_configured_returns_empty(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config", lambda: {"gpu_split_indices": None})
        assert gpu_split_shortfall(10_000_000_000) == []

    def test_equal_split_both_devices_sufficient_returns_empty(self, monkeypatch):
        # 4 GB required, PINNED equal 50/50 split -> 2 GB needed per device;
        # both GPUs (2 GB and 14 GB free) have enough.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]})
        assert gpu_split_shortfall(4_000_000_000) == []

    def test_pinned_equal_split_one_device_short_is_flagged(self, monkeypatch):
        # 8 GB required, PINNED equal 50/50 split -> 4 GB needed per device.
        # GPU 0 only has 2 GB free (short by 2 GB); GPU 1's 14 GB free easily
        # covers its 4 GB share. Only GPU 0 should be flagged. (Unpinned, the
        # auto free-proportional split gives GPU 0 a 1 GB share instead and
        # nothing is short - the behaviour this pin opts out of.)
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]})
        result = gpu_split_shortfall(8_000_000_000)
        assert result == [{"index": 0, "needed": 4_000_000_000, "free": 2_000_000_000}]

    def test_asymmetric_ratio_computes_proportional_need(self, monkeypatch):
        # 10 GB required, ratio 1:4 (GPU0:GPU1) -> GPU0 needs 1/5 = 2 GB
        # (exactly its 2 GB free - not short), GPU1 needs 4/5 = 8 GB (well
        # within its 14 GB free) - nothing flagged.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 4.0]})
        assert gpu_split_shortfall(10_000_000_000) == []

    def test_asymmetric_ratio_flags_the_heavier_device_when_short(self, monkeypatch):
        # Same 1:4 ratio, but now GPU1 (the heavier share) is the tight one.
        gpus = [
            {"index": 0, "name": "A", "total": 16_000_000_000, "free": 14_000_000_000},
            {"index": 1, "name": "B", "total": 16_000_000_000, "free": 2_000_000_000},
        ]
        monkeypatch.setattr("localm.discover.list_gpus", lambda: gpus)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 4.0]})
        # 10 GB required -> GPU0 needs 2 GB (has 14 GB, fine), GPU1 needs 8 GB
        # (has only 2 GB free - short by 6 GB).
        result = gpu_split_shortfall(10_000_000_000)
        assert result == [{"index": 1, "needed": 8_000_000_000, "free": 2_000_000_000}]

    def test_both_devices_short_flags_both(self, monkeypatch):
        gpus = [
            {"index": 0, "name": "A", "total": 16_000_000_000, "free": 1_000_000_000},
            {"index": 1, "name": "B", "total": 16_000_000_000, "free": 1_000_000_000},
        ]
        monkeypatch.setattr("localm.discover.list_gpus", lambda: gpus)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
        result = gpu_split_shortfall(8_000_000_000)
        assert {d["index"] for d in result} == {0, 1}

    def test_fewer_than_two_valid_devices_returns_empty(self, monkeypatch):
        """Only one of the two configured indices is currently detected -
        resolve_gpu_split degrades this to "no split" (its own contract), so
        there is no per-device split share to check at all."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS[:1])
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 5]})
        assert gpu_split_shortfall(100_000_000_000) == []

    def test_unmeasurable_device_free_is_skipped_not_flagged(self, monkeypatch):
        """A device with no 'free' key at all (unmeasurable for that specific
        device) cannot be checked - it must be skipped, not treated as a
        false 0-free shortfall."""
        gpus = [
            {"index": 0, "name": "A", "total": 16_000_000_000},   # no "free" key
            {"index": 1, "name": "B", "total": 16_000_000_000, "free": 14_000_000_000},
        ]
        monkeypatch.setattr("localm.discover.list_gpus", lambda: gpus)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
        assert gpu_split_shortfall(20_000_000_000) == []

    def test_config_param_injection_bypasses_load_config(self, monkeypatch):
        """Matches vram_capacity()/apply_gpu_split()'s existing config= convention."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        result = gpu_split_shortfall(
            8_000_000_000,
            config={"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]})
        assert result == [{"index": 0, "needed": 4_000_000_000, "free": 2_000_000_000}]

    # --- Probe-freshness contract ----------------------------------------------
    # A refusal gate never computes a shortfall (nor quotes a stale "free" MB)
    # from a reading list_gpus served frozen after a probe TIMEOUT/BUSY. These pin
    # both halves: a non-OK probe admits without a figure, and a fresh
    # (GPU_PROBE_OK) probe still bites on a genuine per-device shortfall.

    @staticmethod
    def _probe(gpus, status):
        """A faithful list_gpus double: honours return_status like the real probe, so
        the tolerant reader sees a status-capable callable and consumes ``status``
        rather than defaulting a bare stub to GPU_PROBE_OK."""
        def fake(*a, return_status=False, **k):
            return (gpus, status) if return_status else gpus
        return fake

    def test_stale_timeout_probe_does_not_manufacture_a_shortfall(self, monkeypatch):
        """On TIMEOUT list_gpus serves a FROZEN reading; both devices here read
        far too small for the 20 GB ask, so a bare list_gpus plus an always-run
        loop would flag BOTH and quote their stale free. The gate returns []
        and, opt-in, the non-OK status, so no figure it never measured reaches a
        user."""
        gpus = [{"index": 0, "name": "A", "total": 8_000_000_000, "free": 100_000_000},
                {"index": 1, "name": "B", "total": 8_000_000_000, "free": 100_000_000}]
        cfg = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]}
        monkeypatch.setattr("localm.discover.list_gpus",
                            self._probe(gpus, GPU_PROBE_TIMEOUT))
        assert gpu_split_shortfall(20_000_000_000, cfg) == []
        assert gpu_split_shortfall(20_000_000_000, cfg, return_status=True) == (
            [], GPU_PROBE_TIMEOUT)

    def test_busy_probe_also_admits_without_fabricating(self, monkeypatch):
        """BUSY (a concurrent or wedged probe served the last-known-good value) is
        inconclusive exactly like TIMEOUT: admit, no fabricated figure."""
        gpus = [{"index": 0, "name": "A", "total": 8_000_000_000, "free": 100_000_000},
                {"index": 1, "name": "B", "total": 8_000_000_000, "free": 100_000_000}]
        cfg = {"gpu_split_indices": [0, 1]}
        monkeypatch.setattr("localm.discover.list_gpus",
                            self._probe(gpus, GPU_PROBE_BUSY))
        assert gpu_split_shortfall(20_000_000_000, cfg) == []
        assert gpu_split_shortfall(20_000_000_000, cfg, return_status=True) == (
            [], GPU_PROBE_BUSY)

    def test_fresh_probe_still_reports_a_real_per_device_shortfall(self, monkeypatch):
        """Guards against an over-abstain 'fix' (always return []): a FRESH
        (GPU_PROBE_OK) probe that genuinely cannot cover one device's share must still
        flag it, and every returned free is a live figure a caller may quote."""
        gpus = [{"index": 0, "name": "A", "total": 8_000_000_000, "free": 100_000_000},
                {"index": 1, "name": "B", "total": 64_000_000_000, "free": 50_000_000_000}]
        cfg = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]}
        monkeypatch.setattr("localm.discover.list_gpus",
                            self._probe(gpus, GPU_PROBE_OK))
        expected = [{"index": 0, "needed": 10_000_000_000, "free": 100_000_000}]
        assert gpu_split_shortfall(20_000_000_000, cfg) == expected
        assert gpu_split_shortfall(20_000_000_000, cfg, return_status=True) == (
            expected, GPU_PROBE_OK)

    def test_fresh_all_clear_returns_empty_with_ok_status(self, monkeypatch):
        """A fresh probe where every device covers its share: [] AND GPU_PROBE_OK, so
        a status-aware caller can tell this 'verified clear' from a 'could not check'
        (both return [] bare)."""
        monkeypatch.setattr("localm.discover.list_gpus",
                            self._probe(self._GPUS, GPU_PROBE_OK))
        assert gpu_split_shortfall(
            4_000_000_000, {"gpu_split_indices": [0, 1]}, return_status=True) == (
            [], GPU_PROBE_OK)

    def test_no_split_configured_is_conclusive_ok_without_probing(self, monkeypatch):
        """No split configured -> a conclusive ([], OK) that needs no hardware probe:
        list_gpus must not be called at all."""
        def _boom(*a, **k):
            raise AssertionError("list_gpus must not be probed when no split configured")
        monkeypatch.setattr("localm.discover.list_gpus", _boom)
        assert gpu_split_shortfall(
            10_000_000_000, config={"gpu_split_indices": None},
            return_status=True) == ([], GPU_PROBE_OK)

    def test_deadline_is_forwarded_to_the_probe(self, monkeypatch):
        """An off-loop caller passes a longer deadline so a cold driver init that
        overruns the short server cap still yields a FRESH first-load reading rather
        than timing out into the best-effort admit. The value must reach list_gpus."""
        seen = {}

        def fake(*a, return_status=False, deadline=None, **k):
            seen["deadline"] = deadline
            return (self._GPUS, GPU_PROBE_OK) if return_status else self._GPUS

        monkeypatch.setattr("localm.discover.list_gpus", fake)
        gpu_split_shortfall(4_000_000_000, {"gpu_split_indices": [0, 1]},
                            deadline=_GPU_PROBE_CLI_DEADLINE)
        assert seen["deadline"] == _GPU_PROBE_CLI_DEADLINE

    def test_default_deadline_is_not_forced_onto_the_probe(self, monkeypatch):
        """No deadline given -> the kwarg must NOT be passed at all, leaving list_gpus'
        own default cap intact. Forwarding a literal None would override that real
        default and cap the probe at None."""
        seen = {}

        def fake(*a, return_status=False, **k):
            seen["deadline"] = k.get("deadline", "absent")
            return (self._GPUS, GPU_PROBE_OK) if return_status else self._GPUS

        monkeypatch.setattr("localm.discover.list_gpus", fake)
        gpu_split_shortfall(4_000_000_000, {"gpu_split_indices": [0, 1]})
        assert seen["deadline"] == "absent"

    # --- Blindness axis (free_scope): the REFUSE direction ------------------------
    # A FREE_SCOPE_PROCESS reading OVER-states free (total minus only OUR own use),
    # so a device whose blind free is already short is short FOR REAL and refusing
    # on it is correct. These pin that refusal.

    def test_blind_process_scoped_device_that_is_short_is_still_flagged(self, monkeypatch):
        """A PROCESS-scoped (blind) device whose free is already below its share MUST
        still be flagged. Blind OVER-states free (it misses every other process), so
        blind-free < needed implies real-free < needed, so the refusal is sound and
        the gate must not drop it."""
        gpus = [{"index": 0, "name": "A", "total": 16_000_000_000, "free": 100_000_000,
                 "free_scope": discover.FREE_SCOPE_PROCESS},
                {"index": 1, "name": "B", "total": 64_000_000_000, "free": 50_000_000_000,
                 "free_scope": discover.FREE_SCOPE_DEVICE}]
        cfg = {"gpu_split_indices": [0, 1], "gpu_split_ratios": [1.0, 1.0]}
        monkeypatch.setattr("localm.discover.list_gpus", self._probe(gpus, GPU_PROBE_OK))
        assert gpu_split_shortfall(20_000_000_000, cfg) == [
            {"index": 0, "needed": 10_000_000_000, "free": 100_000_000}]

    def test_all_blind_devices_short_are_all_flagged(self, monkeypatch):
        """Every device PROCESS-scoped and short on a FRESH probe: all are flagged. The
        blind tag never suppresses a refusal - each device's real free is at most its
        blind free, so every one of them genuinely cannot cover its share."""
        gpus = [{"index": 0, "name": "A", "total": 16_000_000_000, "free": 100_000_000,
                 "free_scope": discover.FREE_SCOPE_PROCESS},
                {"index": 1, "name": "B", "total": 16_000_000_000, "free": 100_000_000,
                 "free_scope": discover.FREE_SCOPE_PROCESS}]
        cfg = {"gpu_split_indices": [0, 1]}
        monkeypatch.setattr("localm.discover.list_gpus", self._probe(gpus, GPU_PROBE_OK))
        assert {d["index"] for d in gpu_split_shortfall(20_000_000_000, cfg)} == {0, 1}

    def test_blind_device_with_room_is_not_flagged(self, monkeypatch):
        """The mirror: a blind device whose (over-stated) free covers its share is not
        flagged. This is the PERMIT direction, where blindness genuinely bites - the
        board may be full and this cannot see it - but that is undetectable from the
        reading, so the per-device fit check does not pretend otherwise (a permit-side
        caution belongs with the aggregate gate that owns eviction)."""
        gpus = [{"index": 0, "name": "A", "total": 16_000_000_000, "free": 15_000_000_000,
                 "free_scope": discover.FREE_SCOPE_PROCESS},
                {"index": 1, "name": "B", "total": 16_000_000_000, "free": 15_000_000_000,
                 "free_scope": discover.FREE_SCOPE_PROCESS}]
        cfg = {"gpu_split_indices": [0, 1]}
        monkeypatch.setattr("localm.discover.list_gpus", self._probe(gpus, GPU_PROBE_OK))
        assert gpu_split_shortfall(20_000_000_000, cfg) == []


@pytest.mark.usefixtures("_non_vulkan_host")
class TestFreeScope:
    """free_scope: whether a "free" reading counts EVERY process's VRAM, or only the
    calling process's own allocations.

    On Windows + AMD ROCm/HIP, torch.cuda.mem_get_info reports total minus THIS
    process's own allocations and is blind to every other process: a 4 GB torch
    tensor in a CHILD process moves it by exactly 0. localm loads every GGUF in
    an isolated worker SUBPROCESS, so a model's own VRAM lands squarely in that
    blind spot.

    The device-global source is stubbed here so these stay deterministic on any host
    (the real one is AMD/Windows-only); TestFreeScopeRealHardware covers the real
    thing against real hardware.
    """

    def _gpus(self):
        return [{"index": 0, "name": "A", "total": 16_000_000_000, "free": 15_000_000_000}]

    def test_windows_corrects_free_to_device_global_and_tags_device(self, monkeypatch):
        """The whole point: a reading that counted only our own allocations is
        replaced by one that counts the model in the worker (and the game, and
        ComfyUI), and is labelled as such."""
        monkeypatch.setattr(sys, "platform", "win32")
        # Warm: this covers the CORRECTION, not the cold-budget guard
        # (TestFreeScopeColdBudget owns that). Explicit so the two cannot interfere.
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: True)
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes",
                            lambda gpus: {0: 10_000_000_000})
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free"] == 6_000_000_000    # 16G total - 10G really in use
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE

    def test_windows_without_a_source_keeps_number_but_tags_process(self, monkeypatch):
        """When nothing better can answer, a known-process-local figure is not
        passed off as the board's: it is kept and labelled, which is what makes
        /v1/models/unload report the reading as uncertain instead of asserting a
        wrong number as fact."""
        monkeypatch.setattr(sys, "platform", "win32")
        # This test covers the uncorrected-scope FALLBACK path, not
        # raw_reading_is_process_scoped() itself (TestGpuUsageSourceRobustness owns
        # that), so the input is stubbed directly rather than left sensitive to
        # cross-test state.
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped", lambda: True)
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes",
                            lambda gpus: {})
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free"] == 15_000_000_000   # untouched
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_PROCESS

    def test_non_windows_is_untouched_and_device_scoped(self, monkeypatch):
        """On Linux/NVIDIA the driver query is device-global BY DOCUMENTATION (CUDA
        specifies *free as "free according to the OS" and warns another process can
        move it), so it must not be corrected - and the Windows-only source must not
        even be consulted."""
        monkeypatch.setattr(sys, "platform", "linux")

        def _boom(gpus):
            raise AssertionError("must not consult the Windows source off Windows")

        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes", _boom)
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free"] == 15_000_000_000
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE

    def test_source_failure_is_surfaced_as_process_not_crashed(self, monkeypatch):
        """A driver/ctypes failure in the correction must degrade to "we cannot vouch
        for this number", never take down the caller that only wanted a probe."""
        monkeypatch.setattr(sys, "platform", "win32")
        # See test_windows_without_a_source_keeps_number_but_tags_process above:
        # stub the uncorrected-scope input directly.
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped", lambda: True)

        def _boom(gpus):
            raise OSError("atiadlxx exploded")

        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes", _boom)
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_PROCESS

    def test_used_exceeding_total_clamps_to_zero_not_negative(self, monkeypatch):
        """used and total come from different sources, so their difference can land
        outside [0, total]; free must never go negative."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: True)
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes",
                            lambda gpus: {0: 99_000_000_000})
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free"] == 0

    def test_only_mapped_devices_are_corrected(self, monkeypatch):
        """A multi-GPU box where the source can map only one device: the mapped one
        gets truth, the unmapped one is honestly tagged rather than guessed at."""
        monkeypatch.setattr(sys, "platform", "win32")
        # See test_windows_without_a_source_keeps_number_but_tags_process above:
        # stub the uncorrected-scope input directly.
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped", lambda: True)
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: True)
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes",
                            lambda gpus: {1: 2_000_000_000})
        gpus = [
            {"index": 0, "name": "A", "total": 16_000_000_000, "free": 15_000_000_000},
            {"index": 1, "name": "B", "total": 8_000_000_000, "free": 7_000_000_000},
        ]
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_PROCESS
        assert gpus[0]["free"] == 15_000_000_000
        assert gpus[1]["free_scope"] == discover.FREE_SCOPE_DEVICE
        assert gpus[1]["free"] == 6_000_000_000


@pytest.mark.usefixtures("_non_vulkan_host")
class TestVramCapacityFreeScopePropagation:
    """free_scope must travel WITH the number it describes, all the way to the caller
    that decides whether to present it as current fact."""

    _GPUS_DEVICE = [
        {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000,
         "free_scope": "device"},
        {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000,
         "free_scope": "device"},
    ]
    _GPUS_MIXED = [
        {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000,
         "free_scope": "device"},
        {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000,
         "free_scope": "process"},
    ]

    def test_vram_info_propagates_scope(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS_DEVICE)
        monkeypatch.setattr("localm.config.load_config", lambda: {})
        assert vram_info()["free_scope"] == "device"

    def test_split_sum_is_device_only_when_every_device_is(self, monkeypatch):
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS_DEVICE)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 1]})
        assert vram_capacity()["free_scope"] == "device"

    def test_one_process_scoped_device_makes_the_whole_sum_process_scoped(self, monkeypatch):
        """The sum is missing that device's other-process VRAM, so the TOTAL is not a
        whole-board figure either - it must not be laundered into one by summing with
        a device-scoped sibling."""
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS_MIXED)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 1]})
        assert vram_capacity()["free_scope"] == "process"

    def test_untagged_devices_omit_the_key_rather_than_claim_process(self, monkeypatch):
        """No tag means UNKNOWN. Labelling it "process" would assert a blindness we
        never measured, and would break the plain-dict contract the ~28 test files
        patching list_gpus() rely on."""
        gpus = [
            {"index": 0, "name": "A", "total": 24_000_000_000, "free": 20_000_000_000},
            {"index": 1, "name": "B", "total": 12_000_000_000, "free": 10_000_000_000},
        ]
        monkeypatch.setattr("localm.discover.list_gpus", lambda: gpus)
        monkeypatch.setattr("localm.config.load_config",
                            lambda: {"gpu_split_indices": [0, 1]})
        assert vram_capacity() == {"total": 36_000_000_000, "free": 30_000_000_000}


@pytest.mark.integration
class TestFreeScopeRealHardware:
    """The claim that matters, against the REAL driver and REAL hardware: our
    corrected reading tracks VRAM that ANOTHER PROCESS allocates, which the driver's
    own query provably does not.

    Marked integration (needs a real GPU + a real device-global source), so it is
    excluded from the default run. Every test above uses a stub, which cannot
    reproduce this.
    """

    def test_corrected_free_tracks_another_process_where_the_driver_query_cannot(self):
        import subprocess
        import sys as _sys
        import textwrap
        import time

        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA/ROCm device present to measure")
        # This test's SECOND assertion is that the RAW driver query is blind to the
        # child's allocation, which holds only on Windows with an AMD ROCm/HIP torch
        # build. On Linux, or on any NVIDIA/CUDA build, cudaMemGetInfo is
        # device-global, so the run is gated to where the premise holds.
        if _sys.platform != "win32" or not getattr(torch.version, "hip", None):
            pytest.skip("cross-process blindness is measured only on Windows + an "
                        "AMD ROCm/HIP torch build; raw mem_get_info is device-global "
                        "elsewhere, so this experiment does not apply")
        gpus = list_gpus()
        if not gpus or gpus[0].get("free_scope") != "device":
            pytest.skip("no device-global VRAM source on this host, nothing to verify")

        alloc_bytes = 2 * 1024 ** 3
        child_src = textwrap.dedent(
            """
            import sys, torch
            t = torch.empty(int(sys.argv[1]) // 4, dtype=torch.float32, device="cuda:0")
            torch.cuda.synchronize()
            print("READY", flush=True)
            sys.stdin.readline()
            """
        )
        before = list_gpus()[0]["free"]
        raw_before = torch.cuda.mem_get_info(0)[0]
        child = subprocess.Popen(
            [_sys.executable, "-c", child_src, str(alloc_bytes)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            if (child.stdout.readline() or "").strip() != "READY":
                pytest.skip("child could not allocate on this GPU (busy or too small)")
            time.sleep(1.5)
            during = list_gpus()[0]["free"]
            raw_during = torch.cuda.mem_get_info(0)[0]
        finally:
            try:
                child.stdin.write("go\n")
                child.stdin.flush()
                child.wait(timeout=30)
            except Exception:
                child.kill()

        # Our reading SEES the other process (the whole fix), within a tolerance for
        # the child's own context overhead and the driver's MB granularity.
        assert (before - during) > alloc_bytes * 0.8, (
            f"corrected free did not track another process's {alloc_bytes} B "
            f"allocation: {before} -> {during}")
        # ...and the raw driver query still does NOT, so this test is not passing
        # vacuously because the platform quietly started reporting device-global.
        assert abs(raw_before - raw_during) < 64 * 1024 ** 2, (
            "the raw driver query suddenly tracks other processes; the premise of "
            "this fix changed and it should be re-examined, not silently trusted")


@pytest.mark.usefixtures("_non_vulkan_host")
class TestFreeScopeColdBudget:
    """The correction runs INSIDE the deadline-bounded probe, so it spends the same
    budget the driver call already spent.

    Opening the device-global source is a driver init of ~750ms, against a
    ~0.02ms warm read. That cold cost can push a cold probe past its cap, and a
    timed-out probe costs the caller its free reading ENTIRELY (list_gpus serves
    [], vram_info falls to the registry tier, which has no "free" at all). A
    correct number is not worth trading for NO number, so a cold source is
    skipped when the budget is thin. These guard that trade-off in both
    directions.
    """

    def _gpus(self):
        return [{"index": 0, "name": "A", "total": 16_000_000_000, "free": 15_000_000_000}]

    def test_cold_source_is_skipped_when_the_budget_is_thin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        # See TestFreeScope.test_windows_without_a_source_keeps_number_but_tags_process:
        # stub the uncorrected-scope input directly.
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped", lambda: True)
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: False)
        # 0.4s left of the probe's budget: not enough to absorb a ~750ms cold open.
        monkeypatch.setattr(discover, "_probe_deadline_at",
                            discover.time.monotonic() + 0.4)

        def _must_not_run(gpus):
            raise AssertionError("a cold source must not be opened on a thin budget")

        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes", _must_not_run)
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free"] == 15_000_000_000          # untouched
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_PROCESS   # and SAID so

    def test_cold_source_runs_when_the_budget_is_ample(self, monkeypatch):
        """A CLI caller passes the longer _GPU_PROBE_CLI_DEADLINE precisely so a cold
        box can finish; the correction must take that room when it is there (this is
        what keeps `localm doctor` honest on its single, cold probe)."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: False)
        monkeypatch.setattr(discover, "_probe_deadline_at",
                            discover.time.monotonic() + 12.0)
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes",
                            lambda gpus: {0: 10_000_000_000})
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free"] == 6_000_000_000
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE

    def test_warm_source_runs_even_on_a_thin_budget(self, monkeypatch):
        """A warm read is ~0.02ms and cannot overrun anything, so the guard must not
        keep skipping once the source is open - otherwise a 4s-deadline server would
        skip on every probe forever and the fix would silently never engage."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: True)
        monkeypatch.setattr(discover, "_probe_deadline_at",
                            discover.time.monotonic() + 0.01)
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes",
                            lambda gpus: {0: 10_000_000_000})
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free"] == 6_000_000_000
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE

    def test_no_deadline_published_does_not_block_a_cold_open(self, monkeypatch):
        """_probe_deadline_at is None outside a probe (e.g. a direct call). Unknown
        budget must not be read as "no budget", or the correction would never run."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: False)
        monkeypatch.setattr(discover, "_probe_deadline_at", None)
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes",
                            lambda gpus: {0: 10_000_000_000})
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE

    def test_probe_publishes_its_deadline(self, monkeypatch):
        """The guard is only as good as the budget it reads, so the probe must
        actually publish one."""
        seen = {}

        def _probe():
            seen["deadline_at"] = discover._probe_deadline_at
            return [{"index": 0, "name": "A", "total": 1, "free": 1,
                     "free_scope": "device"}]

        monkeypatch.setattr(discover, "_list_gpus_probe", _probe)
        before = discover.time.monotonic()
        discover.list_gpus(deadline=9.0)
        assert seen["deadline_at"] is not None
        # published as "now + deadline", so it lands ~9s out
        assert 8.0 < (seen["deadline_at"] - before) <= 9.5


def test_no_production_caller_passes_a_short_gpu_probe_deadline():
    """Pins the premise http_server.py's free-VRAM-admission-gate comment relies
    on: a joined probe reading is only PROCESS-scoped
    (TestFreeScopeColdBudget.test_cold_source_is_skipped_when_the_
    budget_is_thin above) when its caller's deadline is too thin to absorb the
    ~750ms cold device-global open. _GPU_PROBE_DEADLINE is a single 15.0s value
    used everywhere, so a "thinner" budget can only exist if some production
    caller explicitly passes one.

    An AST scan of every tracked production .py file (not a grep, which would
    also match the string inside a comment or docstring explaining this very
    property) for every call to list_gpus/vram_info/vram_capacity/
    _list_gpus_kw that passes an explicit `deadline=`, asserting each one
    resolves to _GPU_PROBE_DEADLINE/_GPU_PROBE_CLI_DEADLINE (now the same
    value) or is a plain pass-through of the caller's own `deadline` parameter
    - never a shorter literal or an unrelated expression this scan cannot
    account for."""
    import ast
    from pathlib import Path

    root = Path(discover.__file__).resolve().parent
    target_funcs = {"list_gpus", "vram_info", "vram_capacity", "_list_gpus_kw"}
    safe_names = {"_GPU_PROBE_DEADLINE", "_GPU_PROBE_CLI_DEADLINE", "deadline"}
    offenders = []

    def _call_target(node: ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_target(node) not in target_funcs:
                continue
            for kw in node.keywords:
                if kw.arg != "deadline":
                    continue
                v = kw.value
                if isinstance(v, ast.Name) and v.id in safe_names:
                    continue
                if isinstance(v, ast.Attribute) and v.attr in safe_names:
                    continue
                if isinstance(v, ast.Constant) and v.value == discover._GPU_PROBE_DEADLINE:
                    continue
                offenders.append(
                    f"{path.relative_to(root.parent)}:{node.lineno}: "
                    f"deadline={ast.unparse(v)}")

    assert not offenders, (
        "a production call site now passes a deadline this scan cannot prove "
        "is >= _GPU_PROBE_DEADLINE - this reopens the process-scoped-permit "
        "gap http_server.py's free-VRAM-admission-gate comment (near "
        "get_comfy_status) describes as currently closed; update that comment "
        "if this is deliberate:\n" + "\n".join(offenders))


class TestGpuUsageSourceRobustness:
    """gpu_usage.py source-selection and warmth.

    These guard three defects: a transient PDH failure permanently disabling the
    source, source_is_warm() ignoring PDH's cold cost, and the platform detector
    that decides whether an uncorrected reading is known-blind. All are stubbed
    off real hardware so they run on any host.
    """

    def _reset(self, monkeypatch):
        import localm.gpu_usage as gu
        monkeypatch.setattr(gu, "_adl_state", None)
        monkeypatch.setattr(gu, "_pdh_state", None)
        return gu

    def test_transient_pdh_query_failure_does_not_permanently_poison(self, monkeypatch):
        """A momentary query hiccup must be retryable, not a process-lifetime
        disable. Only a genuinely-permanent condition (win32pdh unimportable)
        may set the sticky {}."""
        gu = self._reset(monkeypatch)
        import types

        calls = {"collect": 0}

        fake_pdh = types.SimpleNamespace(
            PERF_DETAIL_WIZARD=0, PDH_FMT_LARGE=0,
            OpenQuery=lambda: object(),
            EnumObjectItems=lambda *a, **k: ([], ["luid_0_0_phys_0"]),
            MakeCounterPath=lambda spec: "path",
            AddEnglishCounter=lambda q, p: object(),
            GetFormattedCounterValue=lambda h, f: (0, 123),
        )

        def _collect_raises(_q):
            calls["collect"] += 1
            raise OSError("transient PDH hiccup")

        fake_pdh.CollectQueryData = _collect_raises
        monkeypatch.setitem(__import__("sys").modules, "win32pdh", fake_pdh)

        assert gu._pdh_adapter_used() == []
        # state is NOT the sticky-disabled {}: the open query survives for a retry.
        assert gu._pdh_state != {}, "a transient query failure permanently poisoned PDH"
        assert gu._pdh_state is not None and gu._pdh_state.get("query") is not None

        # And a retry after the driver recovers actually works.
        fake_pdh.CollectQueryData = lambda _q: None
        assert gu._pdh_adapter_used() == [123]

    def test_win32pdh_unimportable_is_stickily_disabled(self, monkeypatch):
        """The genuinely-permanent case still sticks: no point retrying an import
        that will never succeed."""
        gu = self._reset(monkeypatch)
        monkeypatch.setitem(__import__("sys").modules, "win32pdh", None)  # import -> TypeError
        assert gu._pdh_adapter_used() == []
        assert gu._pdh_state == {}

    def test_source_is_warm_accounts_for_pdh_not_just_adl(self, monkeypatch):
        """On a non-AMD box ADL is proven-unusable ({}) but PDH is the source that
        answers; keying warmth on ADL alone would call the source warm while a cold
        ~887ms PDH open still lies ahead of the deadline."""
        gu = self._reset(monkeypatch)
        # ADL tried and unusable, PDH not yet opened: a cold PDH open is pending,
        # so NOT warm.
        monkeypatch.setattr(gu, "_adl_state", {})
        monkeypatch.setattr(gu, "_pdh_state", None)
        assert gu.source_is_warm() is False
        # PDH now open -> warm.
        monkeypatch.setattr(gu, "_pdh_state", {"query": object()})
        assert gu.source_is_warm() is True
        # ADL usable -> warm regardless of PDH (AMD path never opens PDH).
        monkeypatch.setattr(gu, "_adl_state", {"dll": object()})
        monkeypatch.setattr(gu, "_pdh_state", None)
        assert gu.source_is_warm() is True
        # Nothing tried yet -> cold.
        monkeypatch.setattr(gu, "_adl_state", None)
        monkeypatch.setattr(gu, "_pdh_state", None)
        assert gu.source_is_warm() is False

    def test_raw_reading_is_process_scoped_only_on_windows_amd_hip(self, monkeypatch):
        import localm.gpu_usage as gu
        import sys as _sys

        class _FakeTorch:
            version = SimpleNamespace(hip="7.13", cuda=None)

        # Windows + ROCm/HIP build -> known blind.
        monkeypatch.setattr(gu.sys, "platform", "win32", raising=False)
        monkeypatch.setitem(_sys.modules, "torch", _FakeTorch())
        assert gu.raw_reading_is_process_scoped() is True
        # Windows + CUDA (NVIDIA) build -> device-global by docs, NOT known blind.
        _FakeTorch.version = SimpleNamespace(hip=None, cuda="12.4")
        monkeypatch.setitem(_sys.modules, "torch", _FakeTorch())
        assert gu.raw_reading_is_process_scoped() is False
        # Non-Windows -> never (device-global by docs), even on a HIP build.
        monkeypatch.setattr(gu.sys, "platform", "linux", raising=False)
        _FakeTorch.version = SimpleNamespace(hip="7.13", cuda=None)
        monkeypatch.setitem(_sys.modules, "torch", _FakeTorch())
        assert gu.raw_reading_is_process_scoped() is False

    def test_raw_reading_is_process_scoped_imports_fresh_when_no_probe_is_inflight(
            self, monkeypatch):
        """The common, safe case: nothing else is touching torch right now, so a
        fresh `import torch` is fine - this must NOT be sacrificed for safety
        against the abandoned-probe race below: always returning False when
        torch is not yet imported is wrong on a machine where torch genuinely
        has not been imported by anything else yet, e.g. very early in the
        process.

        Uses a fake `torch` injected via a patched `__import__`, like the sibling
        test above, rather than deleting the REAL torch from sys.modules and
        letting a genuine re-import run: torch's ROCm SDK native library preload
        is not safe to run a second time in the same process once it has already
        succeeded once (WinError 127, "the specified procedure could not be
        found"), which would make this test flaky for reasons unrelated to the
        code path under test."""
        import localm.gpu_usage as gu
        import localm.discover as _discover
        import sys as _sys
        import builtins

        monkeypatch.setattr(gu.sys, "platform", "win32", raising=False)
        monkeypatch.delitem(_sys.modules, "torch", raising=False)
        monkeypatch.setattr(_discover, "_gpu_probe_inflight", False)
        # Pin the known-doomed detector False so this test, whose subject is the
        # inflight-race logic, does not depend on whether a native HIP runtime is
        # resident. TestTorchProbeKnownDoomedSkip covers the detector itself.
        monkeypatch.setattr(_discover, "_torch_gpu_probe_known_doomed",
                            lambda: False)

        class _FakeTorch:
            version = SimpleNamespace(hip="7.13", cuda=None)

        _real_import = builtins.__import__
        attempted = []

        def _fake_import(name, *a, **k):
            if name == "torch":
                attempted.append(name)
                fake = _FakeTorch()
                # setitem, NOT a raw `_sys.modules["torch"] = fake`: monkeypatch
                # removes the fake at teardown. A raw assignment records nothing to
                # restore where torch is not installed, so the fake survives into
                # every later test in the same xdist worker.
                monkeypatch.setitem(_sys.modules, "torch", fake)
                return fake
            return _real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        assert gu.raw_reading_is_process_scoped() is True
        assert attempted == ["torch"], "the safe import must actually have been attempted"

    def test_raw_reading_is_process_scoped_skips_import_while_a_probe_is_inflight(
            self, monkeypatch):
        """discover._list_gpus_probe's background probe thread does its OWN first
        `import torch` and, on a probe timeout, is ABANDONED while still possibly
        mid-import (stuck in native ROCm library preload);
        discover._gpu_probe_inflight stays True for exactly this window
        (documented in _list_gpus_with_status). A fresh `import torch` from a
        DIFFERENT thread while that flag is set blocks on CPython's per-module
        import lock waiting for the abandoned thread, and crashes the whole
        process (Windows fatal exception, STATUS_ENTRYPOINT_NOT_FOUND) rather
        than merely blocking. raw_reading_is_process_scoped() must consult
        discover._gpu_probe_inflight and skip the import entirely while it is
        True, falling back to the conservative "unknown" default.

        This machine's real torch build IS ROCm/HIP (verified: torch.version.hip
        is set), so if the function fell back to a real `import torch` anyway,
        this would wrongly return True - the test only passes if the import was
        genuinely skipped.

        The resident-HIP fallback signal is pinned False so this test keeps its
        one subject (the inflight race guard): with no HIP runtime resident the
        no-torch answer must stay the conservative False. The signal's own
        truth (and the True case) is covered by TestScopeGateAnswersWithoutTorch
        and TestNativeHipRuntimeResident."""
        import localm.gpu_usage as gu
        import localm.discover as _discover
        import sys as _sys

        monkeypatch.setattr(gu.sys, "platform", "win32", raising=False)
        monkeypatch.delitem(_sys.modules, "torch", raising=False)
        monkeypatch.setattr(_discover, "_gpu_probe_inflight", True)
        monkeypatch.setattr(_discover, "native_hip_runtime_resident",
                            lambda: False)

        # Record every import attempt rather than raising from inside the patched
        # __import__: raw_reading_is_process_scoped() has its own broad
        # `except Exception: return False`, which would swallow a raised guard.
        # Recording and asserting afterward cannot be masked that way.
        import builtins
        _real_import = builtins.__import__
        attempted = []

        def _tracking_import(name, *a, **k):
            attempted.append(name)
            return _real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _tracking_import)

        result = gu.raw_reading_is_process_scoped()
        assert "torch" not in attempted, (
            "raw_reading_is_process_scoped() triggered a fresh import of torch "
            "while a GPU probe was in flight - it can race an abandoned probe "
            "thread mid-import and crash the process (STATUS_ENTRYPOINT_NOT_FOUND, "
            "observed live)")
        assert result is False

    def test_raw_reading_skips_fresh_import_when_known_doomed(self, monkeypatch):
        """The SECOND live site of the known-doomed torch import (the first is
        discover._list_gpus_probe): with the bundled HIP runtime resident and
        torch not yet imported - e.g. the GGUF worker's sizing gate
        (_sizing._device_global_free_bytes) or _apply_device_global_free
        called with no torch loaded - a fresh `import torch` can only fault
        with the same noisy STATUS_ENTRYPOINT_NOT_FOUND stderr trace and land
        in this function's own except anyway. It must consult
        discover._torch_gpu_probe_known_doomed() and never start the import.
        The detector is pinned True (its truth table has its own tests in
        TestTorchProbeKnownDoomedSkip) and the resident-HIP fallback signal
        pinned False, so the answer here must stay the conservative False;
        the True side of the fallback is TestScopeGateAnswersWithoutTorch's
        subject. Attempts are recorded, not raised on, per the sibling tests'
        rationale."""
        import builtins
        import sys as _sys

        import localm.discover as _discover
        import localm.gpu_usage as gu

        monkeypatch.setattr(gu.sys, "platform", "win32", raising=False)
        monkeypatch.delitem(_sys.modules, "torch", raising=False)
        monkeypatch.setattr(_discover, "_gpu_probe_inflight", False)
        monkeypatch.setattr(_discover, "_torch_gpu_probe_known_doomed",
                            lambda: True)
        monkeypatch.setattr(_discover, "native_hip_runtime_resident",
                            lambda: False)

        _real_import = builtins.__import__
        attempted = []

        def _tracking_import(name, *a, **k):
            attempted.append(name)
            return _real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _tracking_import)

        result = gu.raw_reading_is_process_scoped()
        assert "torch" not in attempted, (
            "raw_reading_is_process_scoped() started a fresh `import torch` "
            "despite the known-doomed DLL conflict being detected - that "
            "attempt faults with a 'Windows fatal exception' trace on every "
            "call")
        assert result is False


class TestScopeGateAnswersWithoutTorch:
    """raw_reading_is_process_scoped() must answer from the resident-HIP-runtime
    signal - never a blanket False - whenever torch cannot be consulted (a
    mid-flight probe, the known-doomed DLL conflict, or torch simply not
    installed). The torch-less case is the GGUF WORKER, the process that makes
    the mid-generation context-grow sizing decision: a blanket False keeps the
    device-global correction permanently dead there on the known-blind platform.
    The blindness belongs to the HIP runtime, which torch and the bundled ggml
    query both read (see gpu_usage's module docstring). A fresh `import torch`
    must still never be started on the guarded paths."""

    def _arm(self, monkeypatch, *, inflight, known_doomed, hip_resident):
        import builtins
        import sys as _sys

        import localm.discover as _discover
        import localm.gpu_usage as gu

        monkeypatch.setattr(gu.sys, "platform", "win32", raising=False)
        monkeypatch.delitem(_sys.modules, "torch", raising=False)
        monkeypatch.setattr(_discover, "_gpu_probe_inflight", inflight)
        monkeypatch.setattr(_discover, "_torch_gpu_probe_known_doomed",
                            lambda: known_doomed)
        monkeypatch.setattr(_discover, "native_hip_runtime_resident",
                            lambda: hip_resident)

        real_import = builtins.__import__
        attempted = []

        def _tracking_import(name, *a, **k):
            if name == "torch":
                attempted.append(name)
                raise ImportError("torch is not installed in this scenario")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _tracking_import)
        return gu, attempted

    def test_known_doomed_with_resident_hip_answers_true_without_import(
            self, monkeypatch, caplog):
        """The GGUF-worker case: the doomed import is skipped AND the answer is
        the truthful True, surfaced at debug, never a blanket False that would
        silently disable the correction."""
        gu, attempted = self._arm(monkeypatch, inflight=False,
                                  known_doomed=True, hip_resident=True)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            result = gu.raw_reading_is_process_scoped()
        assert result is True
        assert "torch" not in attempted
        assert "torch is not consultable" in caplog.text

    def test_inflight_probe_with_resident_hip_answers_true_without_import(
            self, monkeypatch):
        gu, attempted = self._arm(monkeypatch, inflight=True,
                                  known_doomed=False, hip_resident=True)
        assert gu.raw_reading_is_process_scoped() is True
        assert "torch" not in attempted

    def test_torch_import_failure_with_resident_hip_answers_true(
            self, monkeypatch):
        """A GGUF-only install (no torch at all) on the HIP build: the import
        legitimately fails, and the resident runtime still answers True."""
        gu, attempted = self._arm(monkeypatch, inflight=False,
                                  known_doomed=False, hip_resident=True)
        assert gu.raw_reading_is_process_scoped() is True
        assert attempted == ["torch"], "the permitted import must be attempted"

    def test_torch_import_failure_without_resident_hip_stays_false(
            self, monkeypatch):
        """No torch AND no resident HIP runtime (a vulkan/cpu build's worker, a
        torch-less NVIDIA box): no measured blindness to assert - False."""
        gu, attempted = self._arm(monkeypatch, inflight=False,
                                  known_doomed=False, hip_resident=False)
        assert gu.raw_reading_is_process_scoped() is False

    def test_torch_pci_bus_never_starts_a_doomed_import(self, monkeypatch):
        """Opening the gate in the worker makes _torch_pci_bus the NEXT
        potential doomed-import site (device_global_used_bytes calls it for the
        ADL mapping): it must decline, not fault with the 0xc0000139 trace."""
        import builtins
        import sys as _sys

        import localm.discover as _discover
        import localm.gpu_usage as gu

        monkeypatch.delitem(_sys.modules, "torch", raising=False)
        monkeypatch.setattr(_discover, "_gpu_probe_inflight", False)
        monkeypatch.setattr(_discover, "_torch_gpu_probe_known_doomed",
                            lambda: True)

        real_import = builtins.__import__
        attempted = []

        def _tracking_import(name, *a, **k):
            if name == "torch":
                attempted.append(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _tracking_import)

        assert gu._torch_pci_bus(0) is None
        assert "torch" not in attempted, (
            "_torch_pci_bus started a fresh `import torch` despite the "
            "known-doomed DLL conflict")


class TestDeviceGlobalUsedTorchlessAdlMapping:
    """device_global_used_bytes() must still map ADL's device-global figure when
    torch cannot supply a PCI bus id (the GGUF worker), via the unambiguous
    single-AMD-adapter + single-requested-GPU rule - and must NOT apply that
    rule where the raw reading is not known-blind (an NVIDIA dGPU next to an
    idle AMD iGPU would otherwise get the WRONG adapter's usage subtracted from
    an already device-global reading). With torch unmappable the PDH fallback
    can also decline (its LUID list shows TWO adapter instances on a real box),
    leaving the opened gate with no source at all, so this rule is what makes
    the worker correction operate."""

    def _arm(self, monkeypatch, *, by_bus, blind, gpus_n=1, name=None):
        import localm.gpu_usage as gu

        monkeypatch.setattr(gu.sys, "platform", "win32", raising=False)
        monkeypatch.setattr(gu, "_adl_used_by_bus", lambda: dict(by_bus))
        monkeypatch.setattr(gu, "_torch_pci_bus", lambda index: None)
        monkeypatch.setattr(gu, "raw_reading_is_process_scoped", lambda: blind)
        monkeypatch.setattr(gu, "_pdh_adapter_used", lambda: [])
        gpus = [{"index": i, "total": 16_000_000_000} for i in range(gpus_n)]
        if name is not None:
            for g in gpus:
                g["name"] = name
        return gu, gpus

    def test_single_amd_adapter_maps_without_torch_on_blind_platform(
            self, monkeypatch):
        gu, gpus = self._arm(monkeypatch, by_bus={45: 2_900_000_000}, blind=True)
        assert gu.device_global_used_bytes(gpus) == {0: 2_900_000_000}

    def test_torchless_amd_named_gpu_maps_when_signal_is_false(
            self, monkeypatch):
        """The torch-less authorisation path: raw_reading_is_process_scoped() is
        legitimately False in a torch-less process where no HIP runtime is
        resident (GGUF loads out-of-process), yet the detected GPU is an AMD
        card. The single-adapter ADL pairing must still fire, recognised by
        the GPU name, so a torch-less build gets a real device-global figure
        instead of nothing. This is the gate discover.vram_info's registry tier
        relies on to recover a device-global free on a GGUF-only install (where
        list_gpus() is empty); the meter fix itself lives in that wiring, not
        here."""
        gu, gpus = self._arm(monkeypatch, by_bus={45: 2_900_000_000}, blind=False,
                             name="AMD Radeon RX 6900 XT")
        assert gu.device_global_used_bytes(gpus) == {0: 2_900_000_000}

    def test_rule_does_not_fire_for_a_non_amd_gpu_when_not_blind(
            self, monkeypatch):
        """The NVIDIA+iGPU safety case: an NVIDIA dGPU is the single detected GPU
        while ADL reports the idle AMD iGPU. Neither signal authorises the
        pairing (not measured-blind, and the detected GPU is not AMD), so report
        nothing rather than subtract the iGPU's usage from the NVIDIA card's
        already device-global reading."""
        gu, gpus = self._arm(monkeypatch, by_bus={45: 2_900_000_000}, blind=False,
                             name="NVIDIA GeForce RTX 4090")
        assert gu.device_global_used_bytes(gpus) == {}

    def test_rule_does_not_fire_with_no_gpu_name_when_not_blind(
            self, monkeypatch):
        """An unrecognised GPU (no name) is not paired on the name signal alone -
        the safe default is to decline, exactly as before the name gate existed
        when the process-scoped signal is also False."""
        gu, gpus = self._arm(monkeypatch, by_bus={45: 2_900_000_000}, blind=False)
        assert gu.device_global_used_bytes(gpus) == {}

    def test_rule_does_not_fire_with_two_amd_adapters(self, monkeypatch):
        gu, gpus = self._arm(
            monkeypatch, by_bus={45: 2_900_000_000, 3: 100}, blind=True)
        assert gu.device_global_used_bytes(gpus) == {}

    def test_rule_does_not_fire_for_two_requested_gpus(self, monkeypatch):
        gu, gpus = self._arm(monkeypatch, by_bus={45: 2_900_000_000}, blind=True,
                             gpus_n=2)
        assert gu.device_global_used_bytes(gpus) == {}

    def test_exact_torch_bus_mapping_still_wins(self, monkeypatch):
        """With a usable pci_bus_id the exact pairing is used, exactly as
        before - the torch-less rule is a fallback, not a replacement."""
        gu, gpus = self._arm(monkeypatch, by_bus={45: 2_900_000_000}, blind=True)
        monkeypatch.setattr(gu, "_torch_pci_bus", lambda index: 45)
        assert gu.device_global_used_bytes(gpus) == {0: 2_900_000_000}

    def test_rule_does_not_fire_when_torch_bus_contradicts(self, monkeypatch):
        """An ANSWERED bus that matches no ADL adapter is an affirmative
        mismatch (or a degraded ADL view - a 2-GPU box can transiently report
        only the wrong adapter, since per-adapter query failures are dropped):
        the fallback is strictly for no-bus-AVAILABLE, so this must decline
        exactly as the pre-rule code did, not pair the contradiction away."""
        gu, gpus = self._arm(monkeypatch, by_bus={45: 2_900_000_000}, blind=True)
        monkeypatch.setattr(gu, "_torch_pci_bus", lambda index: 3)
        assert gu.device_global_used_bytes(gpus) == {}


@pytest.mark.usefixtures("_non_vulkan_host")
class TestUncorrectedScopeIsNotAlwaysProcess:
    """discover._apply_device_global_free must NOT tag every uncorrected Windows
    reading FREE_SCOPE_PROCESS. That over-claim puts a spurious 'process-blind'
    note plus vram_reading_uncertain on NVIDIA/Windows, where cudaMemGetInfo is
    device-global by documentation and the number is fine.
    """

    def _gpus(self):
        return [{"index": 0, "name": "A", "total": 16_000_000_000, "free": 15_000_000_000}]

    def test_windows_non_blind_uncorrected_is_device_not_process(self, monkeypatch):
        """No source maps (empty dict), but the raw reading is NOT known-blind
        (NVIDIA/Windows): tag DEVICE, do not fabricate a blindness."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes", lambda gpus: {})
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped", lambda: False)
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE
        assert gpus[0]["free"] == 15_000_000_000

    def test_windows_blind_uncorrected_is_still_process(self, monkeypatch):
        """The known-blind platform (AMD/Windows) still tags PROCESS when it cannot
        correct - the honest signal must survive."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes", lambda gpus: {})
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped", lambda: True)
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_PROCESS

    def test_cold_budget_skip_uses_platform_scope_not_blanket_process(self, monkeypatch):
        """The cold-budget skip must also honour the distinction: a cold-skipped
        NVIDIA/Windows probe is DEVICE (its number is fine), not a spurious PROCESS."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("localm.gpu_usage.source_is_warm", lambda: False)
        monkeypatch.setattr("localm.gpu_usage.raw_reading_is_process_scoped", lambda: False)
        monkeypatch.setattr(discover, "_probe_deadline_at", discover.time.monotonic() + 0.1)

        def _must_not_run(gpus):
            raise AssertionError("cold budget too thin: source must not be opened")

        monkeypatch.setattr("localm.gpu_usage.device_global_used_bytes", _must_not_run)
        gpus = self._gpus()
        discover._apply_device_global_free(gpus)
        assert gpus[0]["free_scope"] == discover.FREE_SCOPE_DEVICE


@pytest.mark.skipif(sys.platform != "win32",
                    reason="patches ctypes.WinDLL, which exists only on Windows; "
                           "monkeypatch.setattr raises AttributeError elsewhere")
class TestAdlLatchRobustness:
    """_adl_open() must distinguish a PERMANENT unusability (no atiadlxx.dll ->
    not an AMD box) from a TRANSIENT one (driver momentarily not answering
    ADL2_Main_Control_Create). Latching the transient case off for the process
    lifetime is the same missing-vs-corrupt collapse the PDH path can make."""

    def test_missing_dll_is_latched_permanently(self, monkeypatch):
        import localm.gpu_usage as gu
        monkeypatch.setattr(gu, "_adl_state", None)

        def _no_dll(_name):
            raise OSError("atiadlxx.dll not found")

        monkeypatch.setattr(gu.ctypes, "WinDLL", _no_dll)
        assert gu._adl_open() == {}
        assert gu._adl_state == {}, "a missing DLL (permanent) must latch off"

    def test_create_failure_is_retryable_not_latched(self, monkeypatch):
        import localm.gpu_usage as gu
        monkeypatch.setattr(gu, "_adl_state", None)

        class _FakeDLL:
            def ADL2_Main_Control_Create(self, *a):
                return 1  # non-OK: Create failed (driver busy) - transient

        monkeypatch.setattr(gu.ctypes, "WinDLL", lambda _name: _FakeDLL())
        assert gu._adl_open() == {}
        assert gu._adl_state is None, (
            "a transient ADL2_Main_Control_Create failure must NOT latch ADL off for "
            "the process lifetime - _adl_state must stay None so the next call retries")

    def test_successful_open_is_cached(self, monkeypatch):
        import localm.gpu_usage as gu
        monkeypatch.setattr(gu, "_adl_state", None)

        class _FakeDLL:
            def ADL2_Main_Control_Create(self, *a):
                return 0  # OK

        monkeypatch.setattr(gu.ctypes, "WinDLL", lambda _name: _FakeDLL())
        state = gu._adl_open()
        assert state and state.get("dll") is not None
        assert gu._adl_state is state, "a successful open must be cached"


# ------------------------------------------------------------------ #
#  The known-doomed torch import skip (native HIP runtime resident)   #
# ------------------------------------------------------------------ #

class TestNativeHipRuntimeResident:
    """discover.native_hip_runtime_resident() - the shared platform signal for
    both the known-doomed torch-import skip and the torch-less blindness answer
    (gpu_usage.raw_reading_is_process_scoped). True exactly on: Windows + the
    native lib loaded in this process + the resolved runtime shipping a HIP
    ggml backend. The glob runs REAL detection over a real directory - only
    the DLL name is faked (same posture as TestTorchProbeKnownDoomedSkip)."""

    def _arm(self, monkeypatch, tmp_path, *, platform="win32",
             native_loaded=True, hip_dll=True):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.native_lib_loaded",
            lambda: native_loaded)
        rt = tmp_path / "native-runtime"
        rt.mkdir()
        (rt / ("ggml-hip.dll" if hip_dll else "ggml-vulkan.dll")).write_bytes(b"")
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
            lambda: rt)

    def test_true_on_resident_hip_build(self, monkeypatch, tmp_path):
        self._arm(monkeypatch, tmp_path)
        assert discover.native_hip_runtime_resident() is True

    @pytest.mark.parametrize("absent", ["platform", "native", "hip"])
    def test_false_when_any_condition_is_absent(
            self, monkeypatch, tmp_path, absent):
        kwargs = {}
        if absent == "platform":
            kwargs["platform"] = "linux"
        elif absent == "native":
            kwargs["native_loaded"] = False
        else:
            kwargs["hip_dll"] = False
        self._arm(monkeypatch, tmp_path, **kwargs)
        assert discover.native_hip_runtime_resident() is False

    def test_check_failure_answers_false(self, monkeypatch, tmp_path):
        """Fail closed: both consumers treat False as 'no special handling'."""
        self._arm(monkeypatch, tmp_path)

        def _boom():
            raise RuntimeError("resolver broke")

        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
            _boom)
        assert discover.native_hip_runtime_resident() is False


class TestTorchProbeKnownDoomedSkip:
    """_list_gpus_probe() must skip its `import torch` attempt AT THE ROOT
    exactly when that import is known-doomed: Windows + llama.cpp's bundled
    HIP runtime already loaded in this process + a ROCm (rocm_sdk) torch
    installed. There the import can never succeed (the runtime's resident
    same-named DLLs break torch's rocm_sdk preload with
    STATUS_ENTRYPOINT_NOT_FOUND), and every attempt prints a "Windows fatal
    exception" faulthandler trace to stderr. See
    discover._torch_gpu_probe_known_doomed's docstring for the root cause and
    the trade-off.

    The skip must be exactly as narrow as the proven conflict: with ANY of
    its narrowing conditions absent (including a torch already resident in
    sys.modules - importing that again is a free cache hit that cannot
    fault) the probe must still attempt torch, because
    the nvidia-smi fallback cannot see AMD devices - a blanket
    native_lib_loaded() skip (like _sizing's, whose fallback IS lossless)
    would cost real enumeration on setups where torch and the native runtime
    coexist (Linux, a CUDA torch, a Vulkan-build runtime).

    Import attempts are RECORDED via a tracking __import__, never raised on:
    the probe's own broad `except Exception` would mask a raising guard and
    pass the test either way (same rationale as the tracking-import tests
    above). A fake torch is served on interception so an attempted import
    falls through the probe harmlessly - re-importing the REAL torch
    in-process is exactly the unsafe operation under test (see the sibling
    tests' docstrings)."""

    def _arm(self, monkeypatch, tmp_path, *, platform="win32",
             native_loaded=True, hip_dll=True, rocm_sdk_installed=True,
             torch_resident=False):
        """Arrange the guard's four conditions (defaults: the doomed combo)
        and return the list that records every intercepted `import torch`."""
        import builtins
        import importlib.util

        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.native_lib_loaded",
            lambda: native_loaded)
        rt = tmp_path / "native-runtime"
        rt.mkdir()
        # The flavor signal runs the REAL glob over a real directory: only the
        # DLL name is faked, not the detection logic.
        (rt / ("ggml-hip.dll" if hip_dll else "ggml-vulkan.dll")).write_bytes(b"")
        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
            lambda: rt)

        real_find_spec = importlib.util.find_spec

        def _fake_find_spec(name, *args, **kwargs):
            if name == "rocm_sdk":
                return object() if rocm_sdk_installed else None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False))
        if torch_resident:
            # The resident stand-down case: a torch already in sys.modules
            # makes the probe's `import torch` a free cache hit, doom-proof
            # by construction, so the guard must not fire.
            monkeypatch.setitem(sys.modules, "torch", fake_torch)
        else:
            monkeypatch.delitem(sys.modules, "torch", raising=False)
        real_import = builtins.__import__
        attempted = []

        def _tracking_import(name, *args, **kwargs):
            if name == "torch":
                attempted.append(name)
                # setitem (auto-restored), never a raw assignment - see the
                # sys.modules note in the tracking-import test above.
                monkeypatch.setitem(sys.modules, "torch", fake_torch)
                return fake_torch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _tracking_import)

        # The COLD path enumerates OUT of process (a cold `import torch` takes the
        # Windows loader lock and blocks every thread in the process - see
        # localm/_torch_gpu_probe.py), so on that path an "attempt" is the child
        # spawn rather than an in-process import. It is recorded identically, and
        # stubbed so no unit test spawns a real torch child. The resident case
        # still goes in-process and is counted by the tracking import above.
        monkeypatch.setattr(
            discover, "_torch_gpus_isolated",
            lambda: (attempted.append("torch"), [])[1])
        return attempted

    def test_skips_torch_on_the_proven_doomed_combo(
            self, monkeypatch, tmp_path, caplog):
        attempted = self._arm(monkeypatch, tmp_path)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            result = discover._list_gpus_probe()
        assert "torch" not in attempted, (
            "_list_gpus_probe attempted `import torch` with the native HIP "
            "runtime resident and a rocm_sdk torch installed - that attempt is "
            "known-doomed (STATUS_ENTRYPOINT_NOT_FOUND) and prints a 'Windows "
            "fatal exception' trace on every probe")
        assert isinstance(result, list)
        # The skip is surfaced, not silent.
        assert "skipping the torch GPU probe" in caplog.text

    @pytest.mark.parametrize(
        "absent", ["platform", "native", "hip", "rocm_sdk", "resident_torch"])
    def test_attempts_torch_when_any_condition_is_absent(
            self, monkeypatch, tmp_path, absent):
        """The trade-off, pinned: the skip fires ONLY on the full proven combo.
        Anything less (most importantly native_lib_loaded alone - the blanket
        guard _sizing can afford but this probe cannot) keeps the torch
        attempt, and with it the enumeration nvidia-smi cannot provide."""
        kwargs = {}
        if absent == "platform":
            kwargs["platform"] = "linux"
        elif absent == "native":
            kwargs["native_loaded"] = False
        elif absent == "hip":
            kwargs["hip_dll"] = False
        elif absent == "rocm_sdk":
            kwargs["rocm_sdk_installed"] = False
        else:
            # resident_torch: every doom condition holds, but torch is already in
            # sys.modules - the import is a free cache hit that cannot run the
            # rocm_sdk preload, so the guard stands down and the resident torch
            # stays in use.
            kwargs["torch_resident"] = True
        attempted = self._arm(monkeypatch, tmp_path, **kwargs)
        discover._list_gpus_probe()
        assert attempted == ["torch"], (
            f"with {absent!r} absent the combination is not the proven-doomed "
            "one, so the probe must still attempt torch enumeration")

    def test_detector_failure_fails_open_to_the_torch_attempt(
            self, monkeypatch, tmp_path):
        """A broken detector must never cost the working path (fail open)."""
        attempted = self._arm(monkeypatch, tmp_path)

        def _boom():
            raise RuntimeError("detector broke")

        monkeypatch.setattr(
            "localm.inference.backends.llamacpp._loader.runtime_binary_dir",
            _boom)
        discover._list_gpus_probe()
        assert attempted == ["torch"]
