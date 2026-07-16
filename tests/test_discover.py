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
# Imported at collection time ON PURPOSE, though the tests below reach it via
# monkeypatch string targets rather than this name. discover imports it lazily
# (inside _apply_device_global_free), so without this it would first execute INSIDE
# TestListGpus's patch.dict(sys.modules, {"torch": ...}) block - and patch.dict, on
# exit, restores sys.modules to its prior contents, EVICTING it while leaving the
# stale module object bound as an attribute of the localm package. monkeypatch
# resolves "localm.gpu_usage.X" via getattr on that package, so it would patch the
# ORPHANED module while the code under test re-imports a fresh one, and every stub
# below would silently no-op against real hardware. Importing it here keeps it in
# sys.modules from the start, so patch.dict has nothing to evict.
from localm import gpu_usage as _gpu_usage_imported_early  # noqa: F401
from localm.discover import (
    DiscoverError, GPU_PROBE_BUSY, GPU_PROBE_OK, GPU_PROBE_TIMEOUT,
    _GPU_PROBE_CLI_DEADLINE, _GPU_PROBE_DEADLINE, _LLAMA_SPLIT_MODE_LAYER,
    _MAX_GPU_SPLIT_INDEX, _TENSOR_SPLIT_FALLBACK_CAPACITY,
    _native_backend_has_vulkan,
    _quant_of, applied_split_device_count, apply_gpu_split, apply_main_gpu,
    fit_label, gpu_split_shortfall,
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
        # Enumeration fields only, NOT dict equality: entries also carry
        # "free_scope", whose value legitimately depends on the real host (a
        # device-global source exists on Windows, and the driver query is already
        # device-global elsewhere). Pinning it here would make this enumeration test
        # pass or fail on the tester's hardware. TestFreeScope below covers that
        # field directly, with the source stubbed, so it stays deterministic.
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
        fake_proc = MagicMock(returncode=0, stdout=csv)
        with patch.dict(sys.modules, {"torch": None}), \
             patch("subprocess.run", return_value=fake_proc):
            gpus = list_gpus()
        assert len(gpus) == 2
        # free_scope "device" is part of this path's contract, and unlike the torch
        # path it is host-independent: nvidia-smi's memory.free is the whole board's
        # across every process, so it needs no correction on any platform.
        assert gpus[0] == {"index": 0, "name": "NVIDIA RTX 4090",
                            "total": 24576 * 1024 ** 2, "free": 20000 * 1024 ** 2,
                            "free_scope": "device"}
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


class TestListGpusSafety:
    """The safe-by-construction guarantee: list_gpus() must never block its
    caller for longer than its deadline even when the GPU driver probe wedges,
    while ALWAYS returning a fresh reading (no stale free-VRAM). This is the
    regression guard for the diagnosed idle-hang root cause (a synchronous,
    untimed torch.cuda probe on the event loop)."""

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
        polls free-VRAM to confirm a native free landed - a stale value there
        would defeat the AUDIT-MED-11 over-eviction guard."""
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
        reset and used to write _gpu_last_good afterwards - landing inside whatever
        ran next. That is not hypothetical: a cold ROCm init (~6.5s, see this
        module's own note) overruns the 4s deadline, so this box's REAL card was
        arriving inside later tests that assert a fake or empty reading, failing
        them intermittently (~15% of runs) with 'AMD Radeon RX 6900 XT'.

        Guards the epoch fence in _reset_gpu_probe_cache. Clearing the globals
        alone CANNOT fix this (the write happens after the clear), so a fix that
        only re-clears would still fail this test."""
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
    """A timed-out cold probe must be DISTINGUISHABLE from a genuine empty result
    so a blocking caller can retry / report accurately, instead of misattributing
    a slow driver init to 'no GPU / no torch' (AGENTS.md rule 5). Diagnosed cause:
    the FIRST cold torch.cuda/HIP call initializes the ROCm/CUDA driver (~6.5s
    measured) and overruns the 4s server deadline, so list_gpus() returns [] just
    like a no-torch box - with no way, before this, to tell the two apart."""

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

    def test_cli_deadline_is_more_generous_than_the_server_deadline(self):
        """The blocking-caller deadline must give a cold driver init real room,
        and must be strictly larger than the 4s server-loop cap it exists to
        relax - otherwise `localm gpus` inherits the same premature timeout."""
        assert _GPU_PROBE_CLI_DEADLINE > _GPU_PROBE_DEADLINE
        assert _GPU_PROBE_CLI_DEADLINE >= 10.0


class TestListGpusJoinInflight:
    """A patient off-loop caller must be able to JOIN a probe already in flight
    (opt-in wait_for_inflight), not just get an instant BUSY.

    WHY this is the real fix, and a longer deadline alone is not: on a cold box the
    FIRST probe (typically the GUI's 4s /api/stats heartbeat, off-loaded but at the
    4s cap) holds _gpu_probe_inflight for the entire ~4.6s cold ROCm/CUDA init. A
    model-load probe arriving in that window hits the in-flight guard and returns
    BUSY + last-known-good in ~0.000s WITHOUT probing - the identical short-circuit
    a deadline-15 RETRY hits (measured 0.0000s). So switch_engine's VRAM-fit gate
    saw free=None -> measurable=False -> loaded unchecked, EVEN with a long
    deadline, because it never got to run its own probe. Joining the in-flight
    probe is what actually lets the long deadline pay off in the concurrent case.
    The mutation test below (flag OFF -> still BUSY) proves these are not
    tautological: the join, not the deadline, is what changes the outcome."""

    _GOOD = [{"index": 0, "name": "GPU0", "total": 8, "free": 4}]

    @pytest.fixture(autouse=True)
    def _reset_probe_state(self):
        """These tests deliberately leave an abandoned probe thread in flight
        (the whole point: a caller that overran its deadline). Reset before and
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
    """Pin the active native backend to NON-vulkan (REG-596).

    resolve_main_gpu_index/resolve_gpu_split only cross-check a configured index
    against the detected device list when the active backend is NOT vulkan (on
    vulkan, list_gpus() is structurally blind to the real device list, so the
    index is trusted instead - see _native_backend_has_vulkan). That check reads
    the REAL provisioned runtime dir off disk, so without this pin the classes
    below assert the drop-the-unknown-index behaviour on a host that skips it:
    green on an unprovisioned/HIP box, RED on a vulkan-provisioned one (the
    RECOMMENDED universal build), for the same source.

    The vulkan side of the branch is covered explicitly by
    TestVulkanBackendIndexPassthrough, and the detector itself by
    TestNativeBackendHasVulkan, so pinning here narrows these classes to the
    branch they mean to test rather than hiding the other one. Mirrors the
    has_max_devices pin in TestApplyGpuSplit, which fixes the same class of
    host-state leak.
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
        # LM-DA-017: mirrors resolve_gpu_split's ceiling check - must reject
        # an absurd index even when gpus is unmeasurable (list_gpus() -> []),
        # the exact path the ceiling protects (without it, an index like
        # 500000 would reach ctypes.c_int32 main_gpu unchecked).
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
        # CHK-GPUSPLIT-ALLOC: an absurd index must never reach
        # apply_gpu_split's ctypes allocation, even when gpus is unmeasurable
        # (list_gpus() -> [], the exact path this ceiling protects - without
        # it, a configured index like 500000 would size a 500,001-element
        # tensor_split array before the native loader is ever invoked).
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
        # nvidia-smi): the configured indices cannot be cross-checked, so
        # they pass through rather than discarding an explicit user choice
        # we have no way to disprove (same documented boundary as
        # resolve_main_gpu_index).
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
    """REG-596: the vulkan side of the index-validation branch had NO coverage.

    On the vulkan build, list_gpus() cannot enumerate the real device list at
    all (it only sees torch.cuda / nvidia-smi), so a non-empty but
    vulkan-incomplete list must NOT veto an explicitly configured index -
    GPU-SPLIT-VKINDEX was confirmed live to silently drop a VALID device.
    These pin the backend to vulkan and assert the pass-through contract, so
    the behaviour is tested on every host instead of only on whichever runtime
    happens to be provisioned.
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
    """REG-596: the detector whose host-dependence made the suite host-dependent
    had no tests of its own. It reads the real shipped library set, so pin the
    runtime dir at a tmp_path and assert on the file set."""

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
    GPU number - the fix for the bug where a model too big for one GPU alone
    but that fits split across N configured devices was wrongly refused/badged
    against just one device's capacity."""

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


class TestVramInfoReturnStatus:
    """vram_info(return_status=True) must propagate list_gpus()'s own probe
    status, so a caller reporting a specific VRAM number as CURRENT FACT (not
    just a fit ceiling) can tell a fresh reading from a timed-out/stale one -
    the gap a release-verify pass found in /v1/models/unload's vram_before/
    after_bytes reporting."""

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
        info, status = vram_info(return_status=True)
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
        plain no-kwarg stand-in must never see the new kwarg emitted unless
        they opted in - this is what a release-verify-pass fix accidentally
        broke on the first attempt (TypeError: unexpected keyword argument)."""
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
    GGUF-only box (no torch) and the vulkan build (GPU-SPLIT-VKINDEX)."""

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
        # LOAD-BEARING (GPU-SPLIT-VKINDEX): list_gpus() is Vulkan-blind and reports
        # only device 0, but the loader really tensor_splits across [0, 1]. applied_
        # must say 2 (loader truth) while split_device_count collapses to < 2 (the
        # exact bug this fix closes). MUTATION: give applied_ the by_index re-filter
        # split_device_count uses and it returns 1 here -> this goes RED.
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
        # Graft (blast-radius judge): pin applied_ against apply_gpu_split's REAL
        # gate, not a re-derivation. apply_gpu_split returns non-None iff it actually
        # writes a 2+ device tensor_split; applied_ >= 2 must equal that exactly (it
        # IS meant to be that gate). On the vulkan row this also demonstrates the
        # divergence: split_device_count would say < 2 and disagree with the loader.
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
    in a bug report), never presenting an un-run check as passed (AGENTS.md rule 5).
    This one guard fixes both callers (the embedder AND http_server's chat path)."""

    # A MIXED box where the UN-guarded check WOULD flag both devices (tiny free):
    # this is what makes the mutation (revert the guard) go red instead of passing
    # for the wrong reason.
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
        # MUTATION: revert the vulkan guard -> the tiny-free mixed box flags BOTH
        # devices -> result == [{index 0..}, {index 1..}] -> this assertion RED.
        assert result == []
        info = [r for r in caplog.records
                if r.levelno == logging.INFO and "GPU-SPLIT-VKINDEX" in r.getMessage()]
        # MUTATION: drop the log to debug (or remove it) -> no INFO record -> RED.
        # Pins the honesty graft: the skip must reach a bug report, not hide at debug.
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
    enough for a GGUF-backend load - apply_gpu_split() divides a model by a
    STATIC per-config ratio with no live per-device capacity awareness, so an
    asymmetric split (e.g. another already-loaded model sits on one device
    more than another) can pass the aggregate check while one device's actual
    share is short. This is the per-device gate that catches that case."""

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
        # 4 GB required, equal 50/50 split -> 2 GB needed per device; both
        # GPUs (2 GB and 14 GB free) have enough.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
        assert gpu_split_shortfall(4_000_000_000) == []

    def test_equal_split_one_device_short_is_flagged(self, monkeypatch):
        # 8 GB required, equal 50/50 split -> 4 GB needed per device. GPU 0
        # only has 2 GB free (short by 2 GB); GPU 1's 14 GB free easily covers
        # its 4 GB share. Only GPU 0 should be flagged.
        monkeypatch.setattr("localm.discover.list_gpus", lambda: self._GPUS)
        monkeypatch.setattr(
            "localm.config.load_config",
            lambda: {"gpu_split_indices": [0, 1]})
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
            8_000_000_000, config={"gpu_split_indices": [0, 1]})
        assert result == [{"index": 0, "needed": 4_000_000_000, "free": 2_000_000_000}]

    # --- Probe-freshness contract (AGENTS.md rule 5) ----------------------------
    # A refusal gate must never compute a shortfall (and quote its stale "free" MB)
    # from a reading list_gpus served frozen after a probe TIMEOUT/BUSY. These pin
    # both halves: a non-OK probe admits without fabricating a figure, and a fresh
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
        """THE regression. On TIMEOUT list_gpus serves a FROZEN reading; both devices
        here read far too small for the 20 GB ask, so the OLD code (bare list_gpus +
        always-run loop) flagged BOTH and quoted their stale free. The fix returns []
        and, opt-in, the non-OK status - no figure it never measured reaches a user."""
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


@pytest.mark.usefixtures("_non_vulkan_host")
class TestFreeScope:
    """free_scope: whether a "free" reading counts EVERY process's VRAM, or only the
    calling process's own allocations.

    This exists because, measured on a Windows + AMD ROCm/HIP box,
    torch.cuda.mem_get_info reports total - THIS process's own allocations and is
    blind to every other process: with ~10.5 GB genuinely in use it reported 0.14 GB,
    and a plain 4 GB torch tensor in a CHILD process moved it by exactly 0 while an
    OS counter tracked it exactly. localm loads every GGUF in an isolated worker
    SUBPROCESS, so a model's own VRAM lands squarely in that blind spot. See
    dev-notes/vram-cross-process-blindness.md.

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
        """Rule 5: when nothing better can answer we do NOT silently pass a
        known-process-local figure off as the board's - we keep it and say so, which
        is what makes /v1/models/unload report the reading as uncertain instead of
        asserting a wrong number as fact."""
        monkeypatch.setattr(sys, "platform", "win32")
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
    excluded from the default run. Without it, every test above could pass while the
    actual defect - the one a stub cannot reproduce - shipped unfixed.
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
        # child's allocation - which is only true on the measured-blind platform
        # (Windows + an AMD ROCm/HIP torch build). On Linux, or on any NVIDIA/CUDA
        # build, cudaMemGetInfo is device-global, so the raw query DOES see the child
        # and that assertion would spuriously FAIL rather than the test being
        # inapplicable. Gate the run to where the premise actually holds.
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

    Opening the device-global source is a driver init MEASURED at ~750ms, against a
    ~0.02ms warm read. Measured live, that cold cost pushed cold probes from a
    comfortable 2.9-3.5s to 3.6-4.0s against the 4.0s cap and started timing them
    out - and a timed-out probe costs the caller its free reading ENTIRELY (list_gpus
    serves [], vram_info falls to the registry tier, which has no "free" at all). A
    correct number is not worth trading for NO number, so a cold source is skipped
    when the budget is thin. These guard that trade-off in both directions.
    """

    def _gpus(self):
        return [{"index": 0, "name": "A", "total": 16_000_000_000, "free": 15_000_000_000}]

    def test_cold_source_is_skipped_when_the_budget_is_thin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
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


class TestGpuUsageSourceRobustness:
    """gpu_usage.py source-selection and warmth, post-#697 review.

    These guard three defects a review found in the merged fix: a transient PDH
    failure permanently disabling the source, source_is_warm() ignoring PDH's cold
    cost, and the platform detector that decides whether an uncorrected reading is
    known-blind. All are stubbed off real hardware so they run on any host.
    """

    def _reset(self, monkeypatch):
        import localm.gpu_usage as gu
        monkeypatch.setattr(gu, "_adl_state", None)
        monkeypatch.setattr(gu, "_pdh_state", None)
        return gu

    def test_transient_pdh_query_failure_does_not_permanently_poison(self, monkeypatch):
        """A momentary query hiccup must be retryable, not a process-lifetime disable
        (rule 5 missing-vs-corrupt). Only a genuinely-permanent condition
        (win32pdh unimportable) may set the sticky {}."""
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
        # The killer assertion: state is NOT the sticky-disabled {} - the open query
        # survives for a retry.
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
        # ADL tried and unusable, PDH not yet opened: a cold PDH open is pending, so
        # NOT warm (the bug returned True here).
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


@pytest.mark.usefixtures("_non_vulkan_host")
class TestUncorrectedScopeIsNotAlwaysProcess:
    """discover._apply_device_global_free must NOT tag every uncorrected Windows
    reading FREE_SCOPE_PROCESS. That over-claim (the pre-review behaviour) put a
    spurious 'process-blind' note + vram_reading_uncertain on NVIDIA/Windows, where
    cudaMemGetInfo is device-global by documentation and the number is actually fine.
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
