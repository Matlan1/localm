# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the localm MCP server plugin (plugins/mcpserver)."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.mcpserver.server import (
    EngineCache,
    MCPStdioServer,
    build_tools,
)

# A UNC path at a non-routable RFC5737 documentation address (TEST-NET-1), so
# even a total failure of the fix cannot dial a real host.
_UNC = r"\\192.0.2.1\share"
_UNC_FWD = "//192.0.2.1/share"
_DEVICE = r"\\.\PhysicalDrive0"


def _is_unc_or_device(s: str) -> bool:
    """Kept identical to pathsafe.is_unc_or_device_path's forbidden-prefix
    check: judged by Windows rules on every host, not gated on os.name - the
    client-supplied strings these guards cover (pull_model's `repo`,
    run_coder_task's `cwd`, generate_image's `output_path`) are refused
    unconditionally, unlike the local-path (`reject_unsafe_path_string`)
    policy which only checks `//` on Windows."""
    return s[:2] in ("\\\\", "//", "\\/", "/\\")


def _unc_calls(seen) -> list:
    """Filter, don't assert `seen == []`: an unrelated legitimate fs call on
    an ordinary path (made elsewhere during setup or the handler's own logic)
    must not fail a test whose claim is only about the malicious string."""
    return [s for s in seen if _is_unc_or_device(s)]


def _install_fs_spy(monkeypatch, method_name):
    """Record every path string that reaches Path.<method_name>(), and
    hard-fail the call when the string is UNC/device syntax. Same discipline
    as test_admin_fs_routes.py's fs_spy: assert on the absence of the syscall,
    not merely on the returned message, since the defect is the syscall (an
    SMB dial that auto-authenticates), not the response body."""
    seen: list = []
    real = getattr(Path, method_name)

    def spy(self, *a, **kw):
        s = str(self)
        seen.append(s)
        if _is_unc_or_device(s):
            raise AssertionError(
                f"Path.{method_name}() reached the filesystem with a UNC/device "
                f"string: {s!r} - this is the SMB dial (and the net-NTLMv2 "
                "leak), which happens before any status code is chosen")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, method_name, spy)
    return seen


@pytest.fixture
def exists_spy(monkeypatch):
    """pull_model's only fs sink on the UNC-rejection path."""
    return _install_fs_spy(monkeypatch, "exists")


def _stub_engine_factory(model_name):
    engine = MagicMock()
    engine.display_name = model_name
    engine.chat_stream.side_effect = lambda messages, **kw: iter(
        [f"reply-from-{model_name}"])
    engine.embed.return_value = [[0.1, 0.2]]
    # Mirror the real Engine's victim-safety attributes. A bare MagicMock
    # answers every getattr with a truthy Mock, so an idle stub would read as
    # "serving 1 request, mid-unload" to the eviction policy and never be
    # evictable.
    engine.active_requests = 0
    engine.unloading = False
    return engine


GB = 1024 ** 3


def _counting_factory():
    """Engine factory that records every load, so thrash is measurable."""
    loads = []

    def factory(model_name):
        loads.append(model_name)
        return _stub_engine_factory(model_name)

    factory.loads = loads
    return factory


def _resident_cache(default_model="m"):
    return EngineCache(default_model, engine_factory=_counting_factory())


def _fits(free_gb=20):
    """Probe context: fresh reading, plenty of free VRAM."""
    from localm.discover import GPU_PROBE_OK
    return patch("localm.discover.vram_capacity",
                 return_value=({"free": int(free_gb * GB), "total": 24 * GB},
                               GPU_PROBE_OK))


def _too_tight(free_gb=2):
    """Probe context: fresh reading, not enough room for another model."""
    from localm.discover import GPU_PROBE_OK
    return patch("localm.discover.vram_capacity",
                 return_value=({"free": int(free_gb * GB), "total": 24 * GB},
                               GPU_PROBE_OK))


def _probe_inconclusive(free_gb=20):
    """Probe context: the number looks huge but the reading was NOT taken."""
    from localm.discover import GPU_PROBE_TIMEOUT
    return patch("localm.discover.vram_capacity",
                 return_value=({"free": int(free_gb * GB), "total": 24 * GB},
                               GPU_PROBE_TIMEOUT))


def _unmeasurable():
    """Probe context: the box cannot report free VRAM at all."""
    from localm.discover import GPU_PROBE_OK
    return patch("localm.discover.vram_capacity",
                 return_value=({"total": 24 * GB}, GPU_PROBE_OK))


def _sized(gb=4):
    """Every model reports the same known footprint."""
    return patch.object(EngineCache, "_model_required_bytes",
                        lambda self, name: int(gb * GB))


def _no_vram_wait():
    """
    Neutralize the eviction path's before/after VRAM reading.

    _vram_free_reading() calls discover.vram_capacity() too, so a test that
    scripts the ADMIT probe (an iterator, or a raising side_effect) would
    otherwise have that same double consumed/hit by the unload wait as well.
    before=None makes wait_for_vram_release a documented no-op. The unload
    wait's own honesty rules have dedicated tests above.
    """
    return patch("localm.vram._vram_free_reading",
                 return_value=(None, False, None))


def _cfg(**over):
    """Patch load_config with just the residency keys the policy reads."""
    cfg = {"max_resident_models": None, "pinned_models": None}
    cfg.update(over)
    return patch("localm.config.load_config", return_value=cfg)


def _server(default_model="stub-model", enable_images=True):
    engines = EngineCache(default_model=default_model,
                          engine_factory=_stub_engine_factory)
    return MCPStdioServer(build_tools(engines, enable_images=enable_images)), engines


def _req(server, method, params=None, mid=1):
    return server.handle(
        {"jsonrpc": "2.0", "id": mid, "method": method,
         "params": params or {}})


class TestProtocol:
    def test_initialize(self):
        server, _ = _server()
        resp = _req(server, "initialize")
        assert resp["result"]["serverInfo"]["name"] == "localm"
        assert resp["result"]["capabilities"] == {"tools": {}}

    def test_notification_gets_no_reply(self):
        server, _ = _server()
        assert server.handle(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_ping(self):
        server, _ = _server()
        assert _req(server, "ping")["result"] == {}

    def test_unknown_method_errors(self):
        server, _ = _server()
        resp = _req(server, "no/such/method")
        assert resp["error"]["code"] == -32601

    def test_tools_list_includes_all(self):
        server, _ = _server()
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        expected = {
            "chat", "list_models", "system_stats", "search_models",
            "list_model_files", "pull_model", "embed", "generate_image",
            "setup_embeddings", "remove_model", "run_doctor", "list_plugins",
            "install_plugin", "enable_plugin", "disable_plugin", "uninstall_plugin",
            "server_activity"
        }
        assert names == expected

    def test_no_images_flag_hides_tool(self):
        server, _ = _server(enable_images=False)
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "generate_image" not in names


class TestToolAnnotations:
    """MCP tool annotations. Confirmation for destructive tools belongs at the
    CLIENT, so the server DECLARES intent via the standard MCP annotations
    (destructiveHint / readOnlyHint) in tools/list. Without them an MCP client
    has no signal that remove_model / uninstall_plugin delete files."""

    def _annotations_by_name(self, server):
        return {t["name"]: t.get("annotations")
                for t in _req(server, "tools/list")["result"]["tools"]}

    def test_destructive_tools_declare_destructive_hint(self):
        """The primary oracle: the two file-deleting tools carry
        annotations.destructiveHint == true with a human-readable title. Fails
        on pre-fix master, where tools/list emits no annotations at all."""
        server, _ = _server()
        ann = self._annotations_by_name(server)
        for name in ("remove_model", "uninstall_plugin"):
            assert ann[name] is not None, f"{name} advertises no annotations"
            assert ann[name]["destructiveHint"] is True, f"{name} not destructiveHint"
            assert ann[name].get("title"), f"{name} has no human title"

    def test_read_only_tools_declare_read_only_hint(self):
        server, _ = _server()
        ann = self._annotations_by_name(server)
        for name in ("list_models", "list_plugins", "list_model_files",
                     "search_models", "system_stats", "run_doctor"):
            assert ann[name] is not None, f"{name} advertises no annotations"
            assert ann[name]["readOnlyHint"] is True, f"{name} not readOnlyHint"

    def test_destructive_tools_are_not_marked_read_only(self):
        """Negative guard: a destructive tool must never also claim readOnlyHint
        (readOnlyHint true would tell the client destructiveHint is meaningless)."""
        server, _ = _server()
        ann = self._annotations_by_name(server)
        for name in ("remove_model", "uninstall_plugin"):
            assert ann[name].get("readOnlyHint") is not True

    def test_plain_tools_have_no_annotations_key_leaked(self):
        """A tool with no annotations (e.g. chat) must not carry an empty/None
        annotations field - the key is present only when there is a hint."""
        server, _ = _server()
        entries = {t["name"]: t for t in _req(server, "tools/list")["result"]["tools"]}
        assert "annotations" not in entries["chat"]


class TestToolCalls:
    def test_chat_returns_model_output(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "chat", "arguments": {"prompt": "hi"}})
        result = resp["result"]
        assert result["isError"] is False
        assert result["content"][0]["text"] == "reply-from-stub-model"

    def test_chat_model_override(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "chat",
                     "arguments": {"prompt": "hi", "model": "other-model"}})
        assert resp["result"]["content"][0]["text"] == "reply-from-other-model"

    def test_chat_missing_prompt_is_error(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "chat", "arguments": {}})
        assert resp["result"]["isError"] is True

    def test_embed_returns_vectors(self):
        server, _ = _server()
        resp = _req(server, "tools/call",
                    {"name": "embed", "arguments": {"texts": ["a"]}})
        assert json.loads(resp["result"]["content"][0]["text"]) == [[0.1, 0.2]]

    def test_unknown_tool_is_jsonrpc_error(self):
        server, _ = _server()
        resp = _req(server, "tools/call", {"name": "nope", "arguments": {}})
        assert resp["error"]["code"] == -32602

    def test_handler_crash_becomes_tool_error_not_protocol_error(self):
        server, engines = _server()
        server.tools["chat"]["handler"] = MagicMock(side_effect=RuntimeError("boom"))
        resp = _req(server, "tools/call",
                    {"name": "chat", "arguments": {"prompt": "x"}})
        assert "error" not in resp
        assert resp["result"]["isError"] is True
        assert "boom" in resp["result"]["content"][0]["text"]

    def test_list_models_reads_registry(self):
        server, _ = _server()
        with patch("localm.config.load_registry",
                   return_value={"m1": {"path": "x", "source": "local"}}):
            resp = _req(server, "tools/call",
                        {"name": "list_models", "arguments": {}})
        assert "m1" in resp["result"]["content"][0]["text"]

    def test_list_models_survives_malformed_entries(self):
        # One malformed entry does not blank or crash the whole MCP listing: the
        # good model still lists and each bad entry is shown as corrupt.
        server, _ = _server()
        bad_reg = {
            "good": {"path": "x", "source": "local"},
            "bad_str": "oops",
            "bad_null": None,
            "bad_nopath": {"source": "local"},
        }
        with patch("localm.config.load_registry", return_value=bad_reg):
            resp = _req(server, "tools/call",
                        {"name": "list_models", "arguments": {}})
        text = resp["result"]["content"][0]["text"]
        assert resp["result"].get("isError") in (None, False)
        assert "good" in text
        assert text.count("corrupt") == 3


class TestEngineCache:
    def test_same_model_reuses_engine(self):
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        assert cache.get(None) is cache.get(None)

    def test_model_switch_unloads_previous(self):
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        first = cache.get("a")
        second = cache.get("b")
        assert first is not second
        first.unload.assert_called_once()

    def test_model_switch_waits_for_vram_release(self):
        """A model switch must poll for the previous model's VRAM to actually
        free before loading the next one (TDR-hang guard, shared with the
        /v1/models/unload endpoint's own use of wait_for_vram_release)."""
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        cache.get("a")
        with patch("localm.discover.vram_info", return_value={"free": 1_000_000_000}), \
             patch("localm.vram.wait_for_vram_release",
                   return_value=(True, 2_000_000_000)) as mock_wait:
            cache.get("b")
        mock_wait.assert_called_once()
        assert mock_wait.call_args.kwargs["before_bytes"] == 1_000_000_000

    def test_vram_unmeasurable_does_not_block_switch(self):
        """When VRAM cannot be read at all (no GPU/torch), the wait is a no-op -
        matches wait_for_vram_release's own before_bytes=None contract."""
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        cache.get("a")
        with patch("localm.discover.vram_info", return_value={}):
            second = cache.get("b")   # must not hang/raise
        assert second is not None

    def test_switch_release_detected_via_combined_split_capacity(self):
        """The switch's before/after free-VRAM delta must be measured against
        discover.vram_capacity() (combined split capacity), not just the single
        main GPU: a model that frees VRAM mostly on a NON-main split device must
        still be detected as released within the real (not mocked)
        wait_for_vram_release poll."""
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()
        GB = 1024 ** 3
        states = iter([
            # before: GPU0 (main) 10GB free, GPU1 5GB free -> combined 15GB
            [{"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB},
             {"index": 1, "name": "B", "total": 16 * GB, "free": 5 * GB}],
            # after: freed mostly on GPU1 - GPU0 unchanged, GPU1 +10GB free
            [{"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB},
             {"index": 1, "name": "B", "total": 16 * GB, "free": 15 * GB}],
        ])
        last = []

        def _list_gpus():
            nonlocal last
            last = next(states, last)
            return last

        logged = []
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        cache.get("a")
        with patch("localm.discover.list_gpus", side_effect=_list_gpus), \
             patch("localm.config.load_config",
                   return_value={**base_cfg, "gpu_split_indices": [0, 1]}), \
             patch("localm.plugins.mcpserver.server._log", logged.append):
            second = cache.get("b")
        assert second is not None
        assert not any("did not rise" in m for m in logged), (
            "combined split free must show a rise (detected quickly, no "
            f"timeout warning) even though the single main GPU (10GB -> "
            f"10GB) alone never would: {logged}")

    def test_process_scoped_no_rise_logs_could_not_confirm_not_did_not_rise(self):
        """On a Windows/AMD process-scoped reading, the model-switch VRAM check
        must NOT log the false 'did not rise'. The reading is blind to the
        previous model's VRAM (it lives in an isolated worker), so a no-rise
        there proves nothing."""
        from localm.config import load_config as real_load_config
        from localm.discover import GPU_PROBE_OK
        GB = 1024 ** 3
        base_cfg = real_load_config()

        def _list_gpus(*, deadline=None, return_status=False):
            served = [{"index": 0, "name": "A", "total": 16 * GB, "free": 10 * GB,
                       "free_scope": "process"}]
            return (served, GPU_PROBE_OK) if return_status else served

        logged = []
        cache = EngineCache("m", engine_factory=_stub_engine_factory)
        cache.get("a")
        with patch("localm.discover.list_gpus", side_effect=_list_gpus), \
             patch("localm.config.load_config",
                   return_value={k: v for k, v in base_cfg.items()
                                 if k != "gpu_split_indices"}), \
             patch("localm.vram.wait_for_vram_release", return_value=(False, 10 * GB)), \
             patch("localm.plugins.mcpserver.server._log", logged.append):
            second = cache.get("b")
        assert second is not None
        assert any("could not confirm" in m for m in logged), (
            f"a fresh but process-scoped no-rise must be reported as unconfirmable: {logged}")
        assert not any("did not rise" in m for m in logged), (
            f"logged a false 'did not rise' on a blind process-scoped reading: {logged}")

    def test_shutdown_unloads_every_resident_engine(self):
        """N resident means N to free: unloading only the most recent one would
        leave the rest holding VRAM past process exit."""
        cache = _resident_cache()
        with _fits(), _sized():
            a, b = cache.get("a"), cache.get("b")
        assert cache.resident == ["a", "b"]
        cache.unload_all()
        a.unload.assert_called_once()
        b.unload.assert_called_once()
        assert cache.resident == []
        assert cache._engine is None

    def test_shutdown_reports_a_failed_unload_instead_of_swallowing_it(self):
        """A native free that failed is what leaves VRAM pinned after exit, so
        shutdown must report it rather than staying silent."""
        cache = _resident_cache()
        with _fits(), _sized():
            a = cache.get("a")
            cache.get("b")
        a.unload.side_effect = RuntimeError("native free blew up")
        logged = []
        with patch("localm.plugins.mcpserver.server._log", logged.append):
            cache.unload_all()          # must not raise
        assert any("failed to unload a at shutdown" in m for m in logged), logged
        assert cache.resident == []

    def test_no_default_falls_back_to_first_registered(self):
        cache = EngineCache(None, engine_factory=_stub_engine_factory)
        with patch("localm.config.load_registry",
                   return_value={"zeta": {}, "alpha": {}}):
            assert cache.resolve_model(None) == "alpha"

    def test_empty_registry_raises_helpfully(self):
        cache = EngineCache(None, engine_factory=_stub_engine_factory)
        with patch("localm.config.load_registry", return_value={}):
            with pytest.raises(ValueError, match="localm pull"):
                cache.resolve_model(None)


class TestEngineCacheMultiResidency:
    """
    Parity with the HTTP server: a second model that provably fits loads
    ALONGSIDE the first instead of evicting it.
    """

    def test_both_models_stay_resident_when_they_fit(self):
        """THE parity case: measurable, sufficient free VRAM -> zero eviction."""
        cache = _resident_cache()
        with _fits(free_gb=20), _sized(gb=4), _cfg():
            a = cache.get("a")
            b = cache.get("b")
        assert a is not b
        a.unload.assert_not_called()
        assert cache.resident == ["a", "b"]
        assert cache._engines["a"] is a and cache._engines["b"] is b

    def test_resident_model_is_returned_without_reloading(self):
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            a = cache.get("a")
            cache.get("b")
            assert cache.get("a") is a
        assert cache._factory.loads == ["a", "b"]

    def test_alternating_requests_do_not_thrash(self):
        """A,B,A,B,A,B on a box with room must load each model exactly once."""
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            for _ in range(3):
                cache.get("a")
                cache.get("b")
        assert cache._factory.loads == ["a", "b"], (
            f"thrashed: {cache._factory.loads}")
        assert cache.resident == ["a", "b"]

    def test_reusing_a_resident_model_updates_recency(self):
        """LRU order decides the victim, so a reuse must count as a use."""
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            cache.get("a")
            cache.get("b")
            cache.get("a")                     # 'a' is now MRU, 'b' the LRU
        assert cache.resident == ["b", "a"]

    def test_second_model_that_does_not_fit_evicts_the_lru_peer(self):
        cache = _resident_cache()
        with _fits(free_gb=20), _sized(gb=4), _cfg():
            a = cache.get("a")
        with _too_tight(free_gb=2), _sized(gb=4), _cfg():
            b = cache.get("b")
        a.unload.assert_called_once()
        assert cache.resident == ["b"]
        assert cache._engines["b"] is b

    def test_eviction_picks_the_lru_and_leaves_the_rest_resident(self):
        cache = _resident_cache()
        with _fits(free_gb=20), _sized(gb=4), _cfg():
            a, b = cache.get("a"), cache.get("b")
        # Room for one more only after something goes: free rises as each
        # victim is freed, so the loop stops after the first eviction.
        frees = iter([2 * GB, 20 * GB])
        from localm.discover import GPU_PROBE_OK
        with patch("localm.discover.vram_capacity",
                   side_effect=lambda *a_, **k: ({"free": next(frees),
                                                  "total": 24 * GB},
                                                 GPU_PROBE_OK)), \
             _no_vram_wait(), _sized(gb=4), _cfg():
            cache.get("c")
        a.unload.assert_called_once()          # 'a' was least recently used
        b.unload.assert_not_called()           # 'b' survives
        assert cache.resident == ["b", "c"]

    def test_eviction_never_targets_the_requested_model(self):
        """Reloading a resident model must never free the model itself."""
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            a = cache.get("a")
        with _too_tight(), _sized(), _cfg():
            again = cache.get("a")
        assert again is a
        a.unload.assert_not_called()
        assert cache._factory.loads == ["a"]

    def test_a_busy_engine_is_not_evicted(self):
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            a, b = cache.get("a"), cache.get("b")
        a.active_requests = 1                  # 'a' is mid-generation
        with _too_tight(), _sized(), _cfg():
            cache.get("c")
        a.unload.assert_not_called()
        b.unload.assert_called_once()

    def test_unmeasurable_vram_falls_back_to_single_resident(self):
        """A box that cannot report free VRAM must behave exactly as before:
        never stack on an unknown, best-effort single-resident."""
        cache = _resident_cache()
        with _unmeasurable(), _sized(), _cfg():
            a = cache.get("a")
            cache.get("b")
        a.unload.assert_called_once()
        assert cache.resident == ["b"]

    def test_inconclusive_probe_falls_back_to_single_resident(self):
        """A stale/timed-out reading that happens to look huge must not admit a
        stacked load - that is the direction that OOMs the driver."""
        cache = _resident_cache()
        with _probe_inconclusive(free_gb=100), _sized(gb=1), _cfg():
            a = cache.get("a")
            cache.get("b")
        a.unload.assert_called_once()
        assert cache.resident == ["b"]

    def test_unknown_footprint_falls_back_to_single_resident(self):
        """An unregistered/unreadable model cannot be proven to fit, so it gets
        the card to itself rather than being assumed free."""
        cache = _resident_cache()
        with _fits(free_gb=100), _cfg(), \
             patch.object(EngineCache, "_model_required_bytes",
                          lambda self, name: None):
            a = cache.get("a")
            cache.get("b")
        a.unload.assert_called_once()

    def test_resident_but_unloaded_engine_still_goes_through_the_vram_gate(self):
        """A constructed-but-unloaded engine holds NO VRAM, so the free-VRAM
        probe cannot see it. Handing it straight back would let the caller's
        chat_stream() load it on top of whatever else is resident with no gate
        at all - a permit-direction hole. pull_model reaches exactly this state:
        it calls get() and nothing else. http_server's fast path carries the
        same `.loaded` check for the same reason."""
        def unloaded_factory(name):
            e = _stub_engine_factory(name)
            e.loaded = False
            return e

        cache = EngineCache("m", engine_factory=unloaded_factory)
        with _fits(), _sized(), _cfg():
            first = cache.get("a")          # pull_model-style: never loaded
        probed = []
        real = EngineCache._fits_alongside
        with _fits(), _sized(), _cfg(), \
             patch.object(EngineCache, "_fits_alongside",
                          lambda self, n, r: (probed.append(n), real(self, n, r))[1]):
            again = cache.get("a")
        assert again is first, "the pulled engine must be reused, not replaced"
        assert probed == ["a"], (
            f"an unloaded resident engine skipped the VRAM gate: probed={probed}")

    def test_loaded_resident_engine_skips_the_gate(self):
        """Guard on the test above: a genuinely loaded model must still be the
        cheap fast path, with no probe at all."""
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            cache.get("a")
        probed = []
        real = EngineCache._fits_alongside
        with _cfg(), patch.object(EngineCache, "_fits_alongside",
                                  lambda self, n, r: (probed.append(n),
                                                      real(self, n, r))[1]):
            cache.get("a")
        assert probed == [], f"a loaded resident model was re-probed: {probed}"

    def test_a_failed_probe_is_not_read_as_headroom(self):
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            a = cache.get("a")
        logged = []
        with patch("localm.discover.vram_capacity",
                   side_effect=RuntimeError("driver wedged")), \
             _no_vram_wait(), _sized(), _cfg(), \
             patch("localm.plugins.mcpserver.server._log", logged.append):
            cache.get("b")
        a.unload.assert_called_once()
        assert any("VRAM probe failed" in m for m in logged), logged


class TestResidencyKnobs:
    def test_cap_forces_eviction_even_when_vram_fits(self):
        """The point of the knob: 'keep at most N', regardless of headroom."""
        cache = _resident_cache()
        with _fits(free_gb=100), _sized(gb=1), _cfg(max_resident_models=1):
            a = cache.get("a")
            cache.get("b")
        a.unload.assert_called_once()
        assert cache.resident == ["b"]

    def test_cap_of_two_keeps_two_and_evicts_the_third(self):
        cache = _resident_cache()
        with _fits(free_gb=100), _sized(gb=1), _cfg(max_resident_models=2):
            a, b = cache.get("a"), cache.get("b")
            assert cache.resident == ["a", "b"]
            cache.get("c")
        a.unload.assert_called_once()
        b.unload.assert_not_called()
        assert cache.resident == ["b", "c"]

    def test_default_config_does_not_cap(self):
        """Defaults must reproduce the unknobbed policy exactly."""
        cache = _resident_cache()
        with _fits(free_gb=100), _sized(gb=1), _cfg():
            cache.get("a")
            cache.get("b")
            cache.get("c")
        assert cache.resident == ["a", "b", "c"]

    def test_pinned_model_is_never_the_victim(self):
        cache = _resident_cache()
        with _fits(), _sized(), _cfg(pinned_models=["a"]):
            a, b = cache.get("a"), cache.get("b")
        with _too_tight(), _sized(), _cfg(pinned_models=["a"]):
            cache.get("c")
        a.unload.assert_not_called()           # pinned, though least recent
        b.unload.assert_called_once()
        assert "a" in cache.resident

    def test_pin_plus_cap_keeps_the_pinned_pair_resident(self):
        """The user-facing ask: 'keep these two', not LRU roulette."""
        cache = _resident_cache()
        knobs = dict(max_resident_models=2, pinned_models=["a", "b"])
        with _fits(free_gb=100), _sized(gb=1), _cfg(**knobs):
            a, b = cache.get("a"), cache.get("b")
            cache.get("c")                     # over cap, but both peers pinned
        a.unload.assert_not_called()
        b.unload.assert_not_called()
        assert set(cache.resident) == {"a", "b", "c"}

    def test_unevictable_load_says_the_policy_was_missed(self):
        """When pins leave nothing evictable the load still proceeds (a stdio
        tool call has no useful 'try later'), but it must SAY the cap was
        missed rather than pretend it held."""
        cache = _resident_cache()
        knobs = dict(max_resident_models=1, pinned_models=["a"])
        logged = []
        with _fits(free_gb=100), _sized(gb=1), _cfg(**knobs), \
             patch("localm.plugins.mcpserver.server._log", logged.append):
            cache.get("a")
            cache.get("b")
        assert any("resident cap" in m and "loading it anyway" in m
                   for m in logged), logged
        assert set(cache.resident) == {"a", "b"}

    def test_invalid_cap_is_ignored_not_coerced(self):
        cache = _resident_cache()
        with _fits(free_gb=100), _sized(gb=1), _cfg(max_resident_models=0):
            cache.get("a")
            cache.get("b")
        assert cache.resident == ["a", "b"]     # ignored, not treated as 1


class TestEngineCacheBackCompat:
    """_engine/_loaded_name predate multi-residency and stay as MRU views."""

    def test_engine_and_loaded_name_track_the_most_recent_model(self):
        cache = _resident_cache()
        with _fits(), _sized(), _cfg():
            cache.get("a")
            b = cache.get("b")
            assert cache._engine is b
            assert cache._loaded_name == "b"
            a = cache.get("a")
            assert cache._engine is a
            assert cache._loaded_name == "a"

    def test_empty_cache_reports_no_engine(self):
        cache = _resident_cache()
        assert cache._engine is None
        assert cache._loaded_name is None


class TestStdioLoop:
    def test_full_round_trip(self):
        """Feed a complete client session through the stdio loop."""
        server, _ = _server()
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {}}),
            json.dumps({"jsonrpc": "2.0",
                        "method": "notifications/initialized"}),
            "this is not json",   # must be skipped, not crash
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "chat",
                                   "arguments": {"prompt": "hello"}}}),
        ]
        stdin = io.StringIO("\n".join(lines) + "\n")
        stdout = io.StringIO()
        server.run_stdio(stdin=stdin, stdout=stdout)

        responses = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert len(responses) == 2          # init + call; notification silent
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2
        assert responses[1]["result"]["content"][0]["text"] == "reply-from-stub-model"

    def test_stdout_is_pure_jsonrpc(self):
        """Every stdout line must parse as JSON - no log contamination."""
        server, _ = _server()
        stdin = io.StringIO(json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
        stdout = io.StringIO()
        server.run_stdio(stdin=stdin, stdout=stdout)
        for line in stdout.getvalue().splitlines():
            json.loads(line)   # raises if anything non-JSON leaked


class TestClientServerIntegration:
    def test_own_client_can_drive_own_server(self, tmp_path):
        """localcoder's MCP client talks to localm's MCP server in-process
        logic via subprocess - the two halves must interoperate."""
        import sys
        import textwrap
        bridge = tmp_path / "bridge.py"
        bridge.write_text(textwrap.dedent("""\
            from unittest.mock import MagicMock
            from localm.plugins.mcpserver.server import (
                EngineCache, MCPStdioServer, build_tools)

            def factory(name):
                e = MagicMock()
                e.chat_stream.side_effect = lambda m, **k: iter(["pong"])
                return e

            engines = EngineCache("stub", engine_factory=factory)
            MCPStdioServer(build_tools(engines, enable_images=False)).run_stdio()
        """), encoding="utf-8")

        from localm.plugins.coder.mcp import MCPServer
        client = MCPServer("self", sys.executable, [str(bridge)])
        try:
            client.start()
            expected = {
                "chat", "list_models", "system_stats", "search_models",
                "list_model_files", "pull_model", "embed",
                "setup_embeddings", "remove_model", "run_doctor", "list_plugins",
                "install_plugin", "enable_plugin", "disable_plugin",
                "uninstall_plugin", "server_activity",
            }
            assert {t["name"] for t in client.tools} == expected
            res = client.call_tool("chat", {"prompt": "ping"})
            assert res.ok
            assert res.output == "pong"
        finally:
            client.stop()


class TestMcpCliGate:
    """The MCP server became an optional plugin (Phase 3): `localm mcp` refuses to
    serve unless the mcp plugin is enabled, but --print-config always works."""

    @pytest.fixture
    def cfg_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "HOME_DIR", tmp_path)
        monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(_cfg, "REGISTRY_FILE", tmp_path / "registry.json")
        return _cfg

    def test_refuses_when_not_installed(self, cfg_env):
        from click.testing import CliRunner
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 1
        assert "not active" in result.output.lower()
        assert "localm plugin install mcp" in result.output

    def test_print_config_works_when_not_installed(self, cfg_env):
        from click.testing import CliRunner
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, ["--print-config"])
        assert result.exit_code == 0
        assert "mcpServers" in result.output
        assert "localm plugin install mcp" in result.output

    def test_serves_when_installed(self, cfg_env, monkeypatch):
        from click.testing import CliRunner
        from localm.plugins.engine import PluginManager
        PluginManager(None).set_installed_state("mcp", True)   # copy store->installed + enable
        called = {}
        import localm.plugins.mcpserver.server as server
        monkeypatch.setattr(server, "serve_stdio",
                            lambda **k: called.setdefault("ok", True))
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 0
        assert called.get("ok") is True

    def test_installed_but_disabled_refuses(self, cfg_env):
        """Installed-but-disabled (the two-axis case) must NOT serve."""
        from click.testing import CliRunner
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(None)
        mgr.set_installed_state("mcp", True)
        mgr.set_enabled_state("mcp", False)                   # installed, disabled
        from localm.plugins.mcpserver.cli import main
        result = CliRunner().invoke(main, [])
        assert result.exit_code == 1
        assert "not active" in result.output.lower()

    def test_mcp_plugin_available_not_installed_by_default(self, cfg_env):
        from localm.plugins.engine import PluginManager
        state = {p["name"]: p for p in PluginManager(None).api_state()["plugins"]}
        assert "mcp" in state                   # in the available catalog
        assert state["mcp"]["installed"] is False
        assert state["mcp"]["available"] is True
        assert state["mcp"]["scope"] == "mcp"

    def test_print_config_uses_os_correct_path(self, cfg_env, monkeypatch):
        """FAC-14: the Claude Desktop path must match the OS, not hardcode %APPDATA%.

        Drives all three OS branches by patching sys.platform (cli.py reads it at
        call time), so the test proves the fix regardless of the host platform -
        the previous version only asserted the host's own branch, which on
        Windows matched the pre-fix hardcoded %APPDATA% too (a weak guard)."""
        import sys
        from click.testing import CliRunner
        from localm.plugins.mcpserver.cli import main
        cases = {
            "win32": "%APPDATA%\\Claude\\claude_desktop_config.json",
            "darwin": "Library/Application Support/Claude/claude_desktop_config.json",
            "linux": "/.config/Claude/claude_desktop_config.json",
        }
        for plat, expected in cases.items():
            monkeypatch.setattr(sys, "platform", plat)
            out = CliRunner().invoke(main, ["--print-config"]).output
            assert expected in out, f"{plat}: expected {expected!r} in output"
        # the macOS path must NOT be the windows one
        monkeypatch.setattr(sys, "platform", "darwin")
        assert "%APPDATA%" not in CliRunner().invoke(main, ["--print-config"]).output


# --------------------------------------------------------------------------- #
#  Robustness: a malformed JSON-RPC payload must not crash the stdio loop      #
# --------------------------------------------------------------------------- #

class TestStdioRobustness:
    def test_batch_array_is_handled_not_crashed(self):
        """A JSON-RPC batch array must be processed element by element."""
        server, _ = _server()
        batch = json.dumps([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])
        stdout = io.StringIO()
        server.run_stdio(stdin=io.StringIO(batch + "\n"), stdout=stdout)
        ids = {json.loads(l)["id"] for l in stdout.getvalue().splitlines()}
        assert ids == {1, 2}

    def test_scalar_and_null_lines_do_not_crash(self):
        """A bare scalar / null parses but is not a request object."""
        server, _ = _server()
        stdout = io.StringIO()
        server.run_stdio(stdin=io.StringIO('123\n"hi"\ntrue\nnull\n'), stdout=stdout)
        responses = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert len(responses) == 4
        assert all(r["error"]["code"] == -32600 for r in responses)

    def test_handle_rejects_non_dict_directly(self):
        server, _ = _server()
        assert server.handle(123)["error"]["code"] == -32600
        assert server.handle([{"x": 1}])["error"]["code"] == -32600


# --------------------------------------------------------------------------- #
#  generate_image safety: stdout purity, privacy sidecar, path confinement     #
# --------------------------------------------------------------------------- #

class TestGenerateImageSafety:
    def _call(self, server, args):
        return server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "generate_image", "arguments": args}})

    @pytest.mark.parametrize("arg_key", ["output_path", "input_image"])
    def test_path_confined_to_home(self, tmp_path, arg_key):
        """An output_path/input_image outside the localm data dir is refused."""
        server, _ = _server()
        outside = str(tmp_path / "evil.png")        # sibling of LOCALM_HOME (.localm)
        with patch("localm.image_gen.comfy.generate_image") as mock_gen:
            r = self._call(server, {"prompt": "x", arg_key: outside})
        assert r["result"]["isError"] is True
        if arg_key == "output_path":
            assert "data dir" in r["result"]["content"][0]["text"]
        mock_gen.assert_not_called()

    @pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
    def test_output_path_unc_rejected_without_touching_the_filesystem(
            self, monkeypatch, bad):
        """Found during the sweep for pull_model's UNC fix: _confine() called
        p.resolve() before checking home-dir containment. A UNC output_path is
        ABSOLUTE on Windows, so it skipped the `home / p` join and resolved
        the raw client string directly - same SMB-dial defect as pull_model's
        `repo`, a different sink. check_input_image (input_image's own
        confinement, in media/paths.py) already carried this guard; _confine
        was the sibling that missed it."""
        seen = _install_fs_spy(monkeypatch, "resolve")
        server, _ = _server()
        with patch("localm.image_gen.comfy.generate_image") as mock_gen:
            r = self._call(server, {"prompt": "x", "output_path": bad})
        assert r["result"]["isError"] is True
        assert bad not in r["result"]["content"][0]["text"]
        mock_gen.assert_not_called()
        assert _unc_calls(seen) == []

    @pytest.mark.parametrize("bad", [
        "somefile.exe:hidden.gguf", "sub/somefile.exe:hidden.gguf",
    ])
    def test_output_path_reserved_characters_are_rejected(self, bad):
        """An NTFS Alternate Data Stream name is rejected: a colon stays
        confined, so a containment check alone still opens a hidden stream
        behind an apparently-empty sibling."""
        server, _ = _server()
        with patch("localm.image_gen.comfy.generate_image") as mock_gen:
            r = self._call(server, {"prompt": "x", "output_path": bad})
        assert r["result"]["isError"] is True
        mock_gen.assert_not_called()

    def test_output_path_alias_does_not_write_through_a_different_file(
            self, monkeypatch):
        """An OS-level short-name alias resolving output_path to a DIFFERENT,
        real sibling already sitting in the data dir stays strictly inside
        it - containment alone would not catch it. Deterministic simulation,
        same technique as test_pathsafe_confined_under.py's alias tests."""
        from localm.config import home_dir
        server, _ = _server()
        home = home_dir()
        home.mkdir(parents=True, exist_ok=True)
        victim = home / "LongExistingFileName.png"
        victim.write_bytes(b"do not overwrite me")
        alias = "LONGEX~1.PNG"
        victim_resolved = victim.resolve()
        real_resolve = Path.resolve

        def fake_resolve(self, *a, **k):
            if self.name == alias:
                return victim_resolved
            return real_resolve(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", fake_resolve)
        with patch("localm.image_gen.comfy.generate_image") as mock_gen:
            r = self._call(server, {"prompt": "x", "output_path": alias})
        assert r["result"]["isError"] is True
        mock_gen.assert_not_called()
        assert victim.read_bytes() == b"do not overwrite me"

    @pytest.mark.parametrize("secret_name",
                             ["auth.key", "auth.json", "sessions.json"])
    def test_input_image_cannot_name_the_credential_store(self, secret_name):
        """An MCP client is usually an LLM steerable by injected content, and
        input_image is READ and then UPLOADED to ComfyUI - which
        sanitize_comfy_url permits to be a LAN or public host over plaintext
        http. The data dir IS the credential store - auth.key is the plaintext
        owner key - so confining the read to the data dir is not enough.

        Named by file, not "some path outside": inside-the-data-dir is the case
        under test."""
        from localm.config import home_dir
        server, _ = _server()
        home = home_dir()
        secret = home / secret_name
        secret.parent.mkdir(parents=True, exist_ok=True)
        # Readable AND image-shaped, so nothing downstream can be credited with
        # the refusal - the path policy has to be what stops it.
        secret.write_bytes(b"\x89PNG\r\n\x1a\nowner-key-material")

        with patch("localm.image_gen.comfy.generate_image") as mock_gen:
            r = self._call(server, {"prompt": "x", "input_image": str(secret)})

        assert r["result"]["isError"] is True, r["result"]
        mock_gen.assert_not_called()
        body = r["result"]["content"][0]["text"]
        assert "uploads" in body or "galleries" in body, body
        # The refusal must not hand the data dir path (and so the OS username)
        # back to the client.
        assert str(home) not in body, body

    def test_input_image_from_the_uploads_inbox_is_accepted(self):
        """The legitimate flow must survive: a file in the upload inbox still
        reaches generate_image. Without this the test above passes just as well
        against a policy that refuses everything."""
        from localm.config import home_dir
        server, _ = _server()
        home = home_dir()
        src = home / "uploads" / "ref.png"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"\x89PNG\r\n\x1a\nMINE")

        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(True, "ok")) as mock_gen:
            r = self._call(server, {"prompt": "x", "input_image": str(src)})

        assert r["result"].get("isError") is not True, r["result"]
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs["input_image"] == src.resolve()

    @pytest.mark.parametrize("mode,expected_write_sidecar", [
        ("privacy", False),
        ("log", True),
    ])
    def test_mode_controls_sidecar(self, monkeypatch, mode, expected_write_sidecar):
        """Privacy mode must pass write_sidecar=False to generate_image."""
        monkeypatch.setenv("LOCALM_MODE", mode)
        server, _ = _server()
        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(True, "ok")) as mock_gen:
            self._call(server, {"prompt": "x"})
        assert mock_gen.call_args.kwargs.get("write_sidecar") is expected_write_sidecar

    def test_privacy_mode_deletes_comfy_output_copy(self, monkeypatch):
        """Privacy mode must also delete ComfyUI's own on-disk output copy,
        which embeds the full prompt and workflow as PNG metadata, not just
        suppress the sidecar."""
        monkeypatch.setenv("LOCALM_MODE", "privacy")
        server, _ = _server()
        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(True, "ok")) as mock_gen:
            self._call(server, {"prompt": "x"})
        assert mock_gen.call_args.kwargs.get("delete_outputs") is True

    def test_logmode_keeps_output_copy(self, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        server, _ = _server()
        with patch("localm.image_gen.comfy.generate_image",
                   return_value=(True, "ok")) as mock_gen:
            self._call(server, {"prompt": "x"})
        assert not mock_gen.call_args.kwargs.get("delete_outputs")

    def test_generate_image_keeps_stdout_clean(self, capsys):
        """Comfy progress output must go to stderr, not the JSON-RPC stdout."""
        server, _ = _server()

        def noisy(*a, **k):
            print("PROGRESS_NOISE_ON_STDOUT")
            return (True, "ok")

        with patch("localm.image_gen.comfy.generate_image", noisy):
            self._call(server, {"prompt": "x"})
        captured = capsys.readouterr()
        assert "PROGRESS_NOISE_ON_STDOUT" not in captured.out
        assert "PROGRESS_NOISE_ON_STDOUT" in captured.err


class TestChatEmbedStdoutSafety:
    """chat/embed/pull_model's own engines.get() call (a fresh model load, e.g.
    the first turn of a new MCP session) must not leak native load-time
    diagnostics onto the JSON-RPC stdout stream. Same class as
    TestGenerateImageSafety above, but the risky call is engines.get() itself
    rather than a separately-patchable module function."""

    def _noisy_engines(self, default_model="stub-model"):
        def noisy_factory(model_name):
            print("NATIVE_LOAD_NOISE_ON_STDOUT")
            return _stub_engine_factory(model_name)
        return EngineCache(default_model=default_model, engine_factory=noisy_factory)

    def test_chat_load_keeps_stdout_clean(self, capsys):
        engines = self._noisy_engines()
        server = MCPStdioServer(build_tools(engines, enable_images=True))
        resp = _req(server, "tools/call",
                    {"name": "chat", "arguments": {"prompt": "hi"}})
        assert resp["result"]["isError"] is False
        captured = capsys.readouterr()
        assert "NATIVE_LOAD_NOISE_ON_STDOUT" not in captured.out
        assert "NATIVE_LOAD_NOISE_ON_STDOUT" in captured.err

    def test_embed_load_keeps_stdout_clean(self, capsys):
        engines = self._noisy_engines()
        server = MCPStdioServer(build_tools(engines, enable_images=True))
        resp = _req(server, "tools/call",
                    {"name": "embed", "arguments": {"texts": ["a"]}})
        assert resp["result"].get("isError") in (None, False)
        captured = capsys.readouterr()
        assert "NATIVE_LOAD_NOISE_ON_STDOUT" not in captured.out
        assert "NATIVE_LOAD_NOISE_ON_STDOUT" in captured.err

    def test_pull_model_post_download_load_keeps_stdout_clean(self, capsys):
        """The download step was already guarded; the load-after-pull step
        (engines.get(name), after registering) was not."""
        engines = self._noisy_engines()
        server = MCPStdioServer(build_tools(engines, enable_images=True))
        with patch("localm.model_manager.pull.pull_model", return_value=True):
            resp = _req(server, "tools/call",
                        {"name": "pull_model",
                         "arguments": {"repo": "org/repo", "name": "newmodel"}})
        assert resp["result"]["isError"] is False
        captured = capsys.readouterr()
        assert "NATIVE_LOAD_NOISE_ON_STDOUT" not in captured.out
        assert "NATIVE_LOAD_NOISE_ON_STDOUT" in captured.err


class TestModelDiscoveryTools:
    """system_stats / search_models / list_model_files / pull_model - always
    advertised (no plugin gate, unlike run_coder_task) since they only need
    core localm functionality."""

    def _call(self, server, name, args):
        return server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": name, "arguments": args}})

    def test_system_stats_returns_live_stats(self):
        server, _ = _server()
        stats = {"cpu": {"percent": 12.0}, "ram": {"used": 1, "total": 2, "percent": 50.0},
                 "vram": {"used": 1, "total": 8, "percent": 12.5}}
        with patch("localm.sysstats.system_stats", return_value=stats):
            r = self._call(server, "system_stats", {})
        assert json.loads(r["result"]["content"][0]["text"]) == stats

    def test_system_stats_waits_for_the_first_vram_reading(self):
        """The MCP tool is a ONE-SHOT call, unlike the GUI's repeating ~2.5s
        poll - it never gets a second chance to see a VRAM reading that lands
        after it already returned, so it must ask sysstats.system_stats to
        wait for the first reading (wait_first_vram=True) rather than
        silently omitting VRAM on a cold first call."""
        server, _ = _server()
        mock_stats = MagicMock(return_value={"cpu": {"percent": 1.0}})
        with patch("localm.sysstats.system_stats", mock_stats):
            self._call(server, "system_stats", {})
        mock_stats.assert_called_once_with(wait_first_vram=True)

    def test_search_models_wraps_hf_search(self):
        server, _ = _server()
        results = [{"id": "bartowski/Qwen2.5-7B-Instruct-GGUF", "downloads": 999,
                   "likes": 10, "updated": "2026-01-01"}]
        with patch("localm.discover.hf_search", return_value=results) as mock_search:
            r = self._call(server, "search_models", {"query": "qwen", "limit": 5})
        assert json.loads(r["result"]["content"][0]["text"]) == results
        mock_search.assert_called_once_with("qwen", limit=5)

    def test_search_models_surfaces_discover_error(self):
        server, _ = _server()
        from localm.discover import DiscoverError
        with patch("localm.discover.hf_search", side_effect=DiscoverError("offline")):
            r = self._call(server, "search_models", {})
        assert r["result"]["isError"] is True
        assert "offline" in r["result"]["content"][0]["text"]

    def test_list_model_files_missing_repo_is_error(self):
        server, _ = _server()
        r = self._call(server, "list_model_files", {})
        assert r["result"]["isError"] is True
        assert "repo" in r["result"]["content"][0]["text"]

    def test_list_model_files_includes_fit_label(self):
        server, _ = _server()
        files = [{"file": "model.Q4_K_M.gguf", "quant": "Q4_K_M",
                 "size_bytes": 4_000_000_000, "n_parts": 1}]
        with patch("localm.discover.hf_gguf_files", return_value=files), \
             patch("localm.discover.vram_info", return_value={"total": 8_000_000_000}), \
             patch("localm.discover.fit_label", return_value="fits") as mock_fit:
            r = self._call(server, "list_model_files", {"repo": "owner/repo"})
        out = json.loads(r["result"]["content"][0]["text"])
        assert out[0]["fit"] == "fits"
        mock_fit.assert_called_once_with(4_000_000_000, 8_000_000_000)

    def test_list_model_files_fit_reflects_combined_split_capacity(self):
        """The MCP list_model_files tool weighs fit against
        discover.vram_capacity()'s COMBINED split capacity, not just
        vram_info()'s single main-GPU number: a file too big for one GPU
        alone but that fits split across a configured 2-GPU split reports
        "fits"."""
        server, _ = _server()
        # need ~= 15e9*1.1 + 1.5e9 = 18e9: exceeds the 16 GB main GPU alone,
        # but fits under 0.85 * the 24 GB combined split.
        files = [{"file": "model.Q8_0.gguf", "quant": "Q8_0",
                 "size_bytes": 15_000_000_000, "n_parts": 1}]
        from localm.config import load_config as real_load_config
        base_cfg = real_load_config()
        with patch("localm.discover.hf_gguf_files", return_value=files), \
             patch("localm.discover.list_gpus", return_value=[
                 {"index": 0, "name": "A", "total": 16_000_000_000, "free": 16_000_000_000},
                 {"index": 1, "name": "B", "total": 8_000_000_000, "free": 8_000_000_000},
             ]), \
             patch("localm.config.load_config",
                   return_value={**base_cfg, "gpu_split_indices": [0, 1]}):
            r = self._call(server, "list_model_files", {"repo": "owner/repo"})
        out = json.loads(r["result"]["content"][0]["text"])
        assert out[0]["fit"] == "fits"

    def test_list_model_files_surfaces_discover_error(self):
        server, _ = _server()
        from localm.discover import DiscoverError
        with patch("localm.discover.hf_gguf_files", side_effect=DiscoverError("not a repo")):
            r = self._call(server, "list_model_files", {"repo": "nope"})
        assert r["result"]["isError"] is True
        assert "not a repo" in r["result"]["content"][0]["text"]

    @pytest.mark.parametrize("args,expected_missing", [
        ({"name": "m"}, "repo"),
        ({"repo": "owner/repo"}, "name"),
    ])
    def test_pull_model_missing_field_is_error(self, args, expected_missing):
        server, _ = _server()
        r = self._call(server, "pull_model", args)
        assert r["result"]["isError"] is True
        assert expected_missing in r["result"]["content"][0]["text"]

    @pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE, _UNC + r"\sub"])
    def test_pull_model_rejects_unc_and_device_repo_without_touching_the_filesystem(
            self, exists_spy, bad):
        server, _ = _server()
        r = self._call(server, "pull_model", {"repo": bad, "name": "m"})
        assert r["result"]["isError"] is True
        assert bad not in r["result"]["content"][0]["text"], (
            "the raw client string must not be echoed back unsanitised")
        assert _unc_calls(exists_spy) == [], (
            "the UNC/device string reached Path.exists() - the whole finding "
            "is that this syscall happens before any status code is chosen")

    def test_pull_model_existing_local_path_still_reports_the_add_message(
            self, exists_spy, tmp_path):
        """Control: an ordinary EXISTING local path is not UNC/device syntax,
        so the fix must not over-reject it - it still falls through to
        Path.exists() and the pre-existing 'run localm add' message."""
        server, _ = _server()
        r = self._call(server, "pull_model", {"repo": str(tmp_path), "name": "m"})
        assert r["result"]["isError"] is True
        assert "localm add" in r["result"]["content"][0]["text"]
        assert str(tmp_path) in exists_spy

    def test_pull_model_nonexistent_local_path_falls_through_to_pull(
            self, exists_spy, tmp_path):
        """Control: an ordinary NON-existent local path is not UNC/device
        syntax either, so it still falls through past the local-add check to
        the normal pull mechanics."""
        server, _ = _server()
        missing = str(tmp_path / "does-not-exist")
        with patch("localm.model_manager.pull.pull_model", return_value=True) as mock_pull:
            r = self._call(server, "pull_model", {"repo": missing, "name": "m"})
        assert r["result"]["isError"] is False
        mock_pull.assert_called_once_with(missing, name="m")
        assert missing in exists_spy

    def test_pull_model_success_loads_by_default(self):
        server, engines = _server()
        with patch("localm.model_manager.pull.pull_model", return_value=True) as mock_pull:
            r = self._call(server, "pull_model",
                           {"repo": "owner/repo", "file": "m.Q4_K_M.gguf", "name": "m"})
        assert r["result"]["isError"] is False
        assert "loaded" in r["result"]["content"][0]["text"]
        mock_pull.assert_called_once_with("owner/repo:m.Q4_K_M.gguf", name="m")
        assert engines._loaded_name == "m"
        # "loaded ... ready to use" must mean an actual backend load, not just
        # cache registration - engines.get() alone leaves the engine
        # resident-but-unloaded.
        engines._engines["m"].load.assert_called_once()

    def test_pull_model_load_false_skips_loading(self):
        # _server() eagerly resolves the default engine once already (the embed-
        # capability probe in build_tools()), so "never loaded" is asserted as
        # "the newly pulled name never became active", not "nothing is loaded".
        server, engines = _server()
        with patch("localm.model_manager.pull.pull_model", return_value=True):
            r = self._call(server, "pull_model",
                           {"repo": "owner/repo", "name": "m", "load": False})
        assert r["result"]["isError"] is False
        assert "not loaded" in r["result"]["content"][0]["text"]
        assert engines._loaded_name != "m"

    def test_pull_model_failure_is_surfaced(self):
        server, _ = _server()
        with patch("localm.model_manager.pull.pull_model", return_value=False):
            r = self._call(server, "pull_model", {"repo": "owner/repo", "name": "m"})
        assert r["result"]["isError"] is True
        assert "pull failed" in r["result"]["content"][0]["text"]

    def test_pull_model_exception_is_surfaced(self):
        server, _ = _server()
        with patch("localm.model_manager.pull.pull_model",
                   side_effect=RuntimeError("network down")):
            r = self._call(server, "pull_model", {"repo": "owner/repo", "name": "m"})
        assert r["result"]["isError"] is True
        assert "network down" in r["result"]["content"][0]["text"]

    def test_pull_model_load_failure_is_distinguished_from_pull_failure(self):
        engines = EngineCache(default_model=None, engine_factory=_stub_engine_factory)
        server = MCPStdioServer(build_tools(engines))
        with patch("localm.model_manager.pull.pull_model", return_value=True), \
             patch.object(engines, "get", side_effect=RuntimeError("no GPU memory")):
            r = self._call(server, "pull_model", {"repo": "owner/repo", "name": "m"})
        assert r["result"]["isError"] is True
        assert "loading it failed" in r["result"]["content"][0]["text"]
        assert "no GPU memory" in r["result"]["content"][0]["text"]

    def test_pull_model_load_step_failure_is_distinguished_from_pull_failure(self):
        """Distinct from the case above, where get() itself raises. Here get()
        succeeds (construction plus cache registration) and the .load() call is
        what fails."""
        def failing_load_factory(name):
            engine = _stub_engine_factory(name)
            engine.load.side_effect = RuntimeError("no GPU memory")
            return engine

        engines = EngineCache(default_model=None, engine_factory=failing_load_factory)
        server = MCPStdioServer(build_tools(engines))
        with patch("localm.model_manager.pull.pull_model", return_value=True):
            r = self._call(server, "pull_model", {"repo": "owner/repo", "name": "m"})
        assert r["result"]["isError"] is True
        assert "loading it failed" in r["result"]["content"][0]["text"]
        assert "no GPU memory" in r["result"]["content"][0]["text"]
        engines._engines["m"].load.assert_called_once()


class TestRunCoderTask:
    """run_coder_task shells out to `localm coder --output-format json` and is
    only advertised when the coder plugin is installed+enabled (mirrors the
    embed tool's capability-gated advertisement)."""

    @pytest.fixture
    def coder_active(self):
        from localm.plugins.engine import PluginManager
        PluginManager(None).set_installed_state("coder", True)

    def _call(self, server, args):
        return server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "run_coder_task", "arguments": args}})

    def test_hidden_when_coder_not_active(self):
        server, _ = _server()
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "run_coder_task" not in names

    def test_listed_when_coder_active(self, coder_active):
        server, _ = _server()
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "run_coder_task" in names

    def test_no_coder_flag_hides_tool_even_when_active(self, coder_active):
        engines = EngineCache(default_model="stub", engine_factory=_stub_engine_factory)
        server = MCPStdioServer(build_tools(engines, enable_coder=False))
        names = {t["name"] for t in _req(server, "tools/list")["result"]["tools"]}
        assert "run_coder_task" not in names

    @pytest.mark.parametrize("missing_field", ["task", "cwd"])
    def test_missing_field_is_error(self, coder_active, tmp_path, missing_field):
        server, _ = _server()
        args = {"task": "do a thing", "cwd": str(tmp_path)}
        del args[missing_field]
        r = self._call(server, args)
        assert r["result"]["isError"] is True
        assert missing_field in r["result"]["content"][0]["text"]

    # This arg dict is complete, so it exercises the existing-but-invalid cwd
    # path rather than the missing-required-field check above.
    def test_nonexistent_cwd_is_error(self, coder_active, tmp_path):
        server, _ = _server()
        r = self._call(server, {"task": "x", "cwd": str(tmp_path / "nope")})
        assert r["result"]["isError"] is True
        assert "directory" in r["result"]["content"][0]["text"]

    @pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
    def test_unc_and_device_cwd_rejected_without_touching_the_filesystem(
            self, coder_active, monkeypatch, bad):
        """Found during the sweep for pull_model's UNC fix: cwd_path.is_dir()
        ran on the client-supplied cwd with no lexical check at all - the same
        SMB-dial defect, a different sink (is_dir() instead of exists())."""
        seen = _install_fs_spy(monkeypatch, "is_dir")
        server, _ = _server()
        r = self._call(server, {"task": "x", "cwd": bad})
        assert r["result"]["isError"] is True
        assert bad not in r["result"]["content"][0]["text"]
        assert _unc_calls(seen) == []

    def test_successful_run_parses_json_payload(self, coder_active, tmp_path):
        # Real `--output-format json` pretty-prints (indent=2, multi-line), so a
        # single-line json.dumps() here would not exercise the same parsing.
        server, _ = _server()
        payload = {"success": True, "response": "done: added type hints",
                   "turns": 3, "total_tokens": 512}
        fake = MagicMock(stdout=json.dumps(payload, indent=2) + "\n", stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake) as mock_run:
            r = self._call(server, {"task": "add type hints", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is False
        assert "done: added type hints" in r["result"]["content"][0]["text"]
        assert "turns=3" in r["result"]["content"][0]["text"]
        cmd = mock_run.call_args.args[0]
        assert cmd[:4] == [sys.executable, "-m", "localm", "coder"]
        assert "add type hints" in cmd
        assert "--cwd" in cmd and str(tmp_path) in cmd
        assert "--output-format" in cmd and "json" in cmd
        assert "--yes" not in cmd            # default off
        # The subprocess's OWN OS cwd must also be the task dir, not just the
        # --cwd flag: if the coder auto-spawns a background server, it identifies
        # "this project" by ITS OWN inherited working directory, so --cwd alone
        # registers the auto-spawned server under the wrong project root.
        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)

    def test_subprocess_env_pins_the_servers_home_and_code(self, coder_active, tmp_path):
        """The coder chain re-resolves BOTH the localm data home and (via
        `-m`'s cwd-first sys.path) the localm PACKAGE from ambient state at
        every process boundary, and this handler runs the child in the TASK's
        directory (the cwd assertion above). A server whose own home came from
        ITS cwd would otherwise hand the chain a different, empty home. The
        handler must pin ITS OWN resolved identity into the child env:
        LOCALM_HOME (same data home), plus PYTHONSAFEPATH and a PYTHONPATH entry
        for its own package root (same code, whatever the task directory
        contains)."""
        import os
        from pathlib import Path
        import localm as _pkg
        from localm.config import home_dir
        server, _ = _server()
        payload = {"success": True, "response": "ok", "turns": 1, "total_tokens": 1}
        fake = MagicMock(stdout=json.dumps(payload, indent=2) + "\n", stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake) as mock_run:
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is False
        env = mock_run.call_args.kwargs["env"]
        assert env["LOCALM_HOME"] == str(home_dir())
        assert env["PYTHONSAFEPATH"] == "1"
        pkg_root = str(Path(_pkg.__file__).resolve().parent.parent)
        assert pkg_root in env.get("PYTHONPATH", "").split(os.pathsep)

    def test_agent_reported_failure_is_surfaced_as_error(self, coder_active, tmp_path):
        server, _ = _server()
        payload = {"success": False, "response": "hit max turns", "turns": 40,
                   "total_tokens": 9001}
        fake = MagicMock(stdout=json.dumps(payload, indent=2) + "\n", stderr="", returncode=1)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is True
        assert "hit max turns" in r["result"]["content"][0]["text"]

    def test_yes_model_max_turns_thread_through_to_cmd(self, coder_active, tmp_path):
        server, _ = _server()
        fake = MagicMock(stdout=json.dumps({"success": True, "response": "ok"}, indent=2) + "\n",
                         stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake) as mock_run:
            self._call(server, {"task": "x", "cwd": str(tmp_path), "model": "qwen2.5-7b",
                                "max_turns": 10, "yes": True})
        cmd = mock_run.call_args.args[0]
        assert "--model" in cmd and "qwen2.5-7b" in cmd
        assert "--max-turns" in cmd and "10" in cmd
        assert "--yes" in cmd

    def test_console_messages_before_json_are_ignored(self, coder_active, tmp_path):
        """Regression guard: a live run against a real model produced exactly
        this shape - console.print() messages (e.g. attaching to a running
        server) print to stdout BEFORE the final --output-format json dump.
        Parsing must find the JSON, not mistake the console text or the bare
        closing brace for the payload."""
        server, _ = _server()
        payload = {"success": True, "response": "created hello.txt", "turns": 2,
                   "total_tokens": 123}
        stdout = (
            "Using the localm already running for /some/project "
            "(port 8642, mode api) - sharing its loaded model.\n"
            + json.dumps(payload, indent=2) + "\n"
        )
        fake = MagicMock(stdout=stdout, stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is False
        assert "created hello.txt" in r["result"]["content"][0]["text"]

    def test_console_messages_after_json_are_ignored(self, coder_active, tmp_path):
        """The mirror of the before-json guard above: with the coder session in
        `--mode full`, "Session transcript saved -> <path>" prints AFTER the
        --output-format json dump, and a successful task must not be reported as
        an error because of that trailing text."""
        server, _ = _server()
        payload = {"success": True, "response": "IDENTITY-FIX-OK", "turns": 1,
                   "total_tokens": 1988}
        stdout = (
            "No server running. Starting one in the background...\n"
            "connected to newly started server at a loopback address\n"
            + json.dumps(payload, indent=2) + "\n"
            + "Session transcript saved ->\n"
            "some/project/.localcoder/sessions/2026-07-22_101521.md\n"
        )
        fake = MagicMock(stdout=stdout, stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is False
        assert "IDENTITY-FIX-OK" in r["result"]["content"][0]["text"]

    def test_no_json_object_in_stdout_falls_back_cleanly(self, coder_active, tmp_path):
        server, _ = _server()
        fake = MagicMock(stdout="some console message, no JSON at all\n",
                         stderr="", returncode=1)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is True
        assert "some console message" in r["result"]["content"][0]["text"]

    def test_timeout_is_reported_not_raised(self, coder_active, tmp_path):
        import subprocess as _sp
        server, _ = _server()
        with patch("localm.plugins.mcpserver.server.subprocess.run",
                   side_effect=_sp.TimeoutExpired(cmd="x", timeout=5)):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path), "timeout_seconds": 5})
        assert r["result"]["isError"] is True
        assert "timed out" in r["result"]["content"][0]["text"]

    def test_non_json_output_falls_back_to_stderr_detail(self, coder_active, tmp_path):
        server, _ = _server()
        fake = MagicMock(stdout="", stderr="coder plugin is not active\n", returncode=1)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake):
            r = self._call(server, {"task": "x", "cwd": str(tmp_path)})
        assert r["result"]["isError"] is True
        assert "coder plugin is not active" in r["result"]["content"][0]["text"]


class TestMcpCliWiring:
    """Regression guard: `localm mcp` must resolve to the real plugin command
    (plugins/mcpserver/cli.py), not a broken hand-rolled duplicate. A duplicate
    in localm/cli/maintenance.py once shadowed it and made `localm mcp` crash
    with an ImportError before it could even print its 'not enabled' message.
    """

    def test_mcp_command_is_the_plugin_command(self):
        from localm.cli import main
        cmd = main.commands.get("mcp")
        assert cmd is not None, "`localm mcp` is not registered"
        # The real command lives in the mcpserver plugin, not in cli.maintenance.
        assert cmd.callback.__module__ == "localm.plugins.mcpserver.cli"

    def test_print_config_works_and_needs_no_active_plugin(self):
        # --print-config is exempt from the active-plugin check, so it works on a
        # fresh install and proves the real (option-bearing) command is wired.
        from click.testing import CliRunner

        from localm.cli import main
        r = CliRunner().invoke(main, ["mcp", "--print-config"])
        assert r.exit_code == 0, r.output
        assert "mcpServers" in r.output and "localm" in r.output
        # the broken duplicate had no such option and never touched winconsole
        assert "winconsole" not in r.output
        assert "unexpected error" not in r.output.lower()


class TestNewToolCalls:
    def test_setup_embeddings(self):
        """The happy path now needs a REAL model identity: over MCP the `model`
        argument must be a known embedding key or an already-registered name, not
        a free-form path, because the client driving it is normally a model acting
        on text it read. "fake-model" was fine when the tool accepted anything;
        it is now refused, so use a known key to exercise the same success path.
        The refusal itself is covered in tests/test_config_admin_gating.py."""
        server, _ = _server()
        with patch("localm.inference.embedder.resolve_embedding_model_path", return_value="fake-path") as mock_resolve:
            resp = _req(server, "tools/call",
                        {"name": "setup_embeddings",
                         "arguments": {"model": "bge-small-en-v1.5"}})
        assert resp["result"]["isError"] is False
        assert "ready" in resp["result"]["content"][0]["text"]
        mock_resolve.assert_called_once()

    def test_remove_model(self):
        server, _ = _server()
        with patch("localm.config.load_registry", return_value={"to-remove": {"path": "x"}}), \
             patch("localm.model_manager.remove_model") as mock_remove:
            resp = _req(server, "tools/call",
                        {"name": "remove_model", "arguments": {"model": "to-remove"}})
        assert resp["result"]["isError"] is False
        assert "removed" in resp["result"]["content"][0]["text"]
        mock_remove.assert_called_once_with("to-remove")

    def test_run_doctor(self):
        server, _ = _server()
        fake_proc = MagicMock(stdout="doctor-ok", stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake_proc) as mock_run:
            resp = _req(server, "tools/call", {"name": "run_doctor", "arguments": {}})
        assert resp["result"]["isError"] is False
        assert "doctor-ok" in resp["result"]["content"][0]["text"]
        mock_run.assert_called_once()

    def test_run_doctor_env_pins_the_servers_home_and_code(self):
        """Same identity pinning as run_coder_task's own env test: a doctor
        child that re-resolves home/code from ambient state reports on the
        WRONG install whenever this server's home came from its cwd (the
        source-checkout `localm mcp` setup)."""
        import os
        from pathlib import Path
        import localm as _pkg
        from localm.config import home_dir
        server, _ = _server()
        fake_proc = MagicMock(stdout="doctor-ok", stderr="", returncode=0)
        with patch("localm.plugins.mcpserver.server.subprocess.run", return_value=fake_proc) as mock_run:
            _req(server, "tools/call", {"name": "run_doctor", "arguments": {}})
        env = mock_run.call_args.kwargs["env"]
        assert env["LOCALM_HOME"] == str(home_dir())
        assert env["PYTHONSAFEPATH"] == "1"
        pkg_root = str(Path(_pkg.__file__).resolve().parent.parent)
        assert pkg_root in env.get("PYTHONPATH", "").split(os.pathsep)

    def test_list_plugins(self):
        server, _ = _server()
        fake_state = {"plugins": [{"name": "fake", "description": "d", "installed": True, "active": True}]}
        with patch("localm.plugins.engine.PluginManager.api_state", return_value=fake_state):
            resp = _req(server, "tools/call", {"name": "list_plugins", "arguments": {}})
        assert resp["result"]["isError"] is False
        assert "fake" in resp["result"]["content"][0]["text"]
        assert "enabled" in resp["result"]["content"][0]["text"]

    def test_install_plugin(self):
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.set_installed_state") as mock_install, \
             patch("localm.plugins.engine.PluginManager.plugin_missing_deps", return_value=False):
            resp = _req(server, "tools/call",
                        {"name": "install_plugin", "arguments": {"plugin": "coder"}})
        assert resp["result"]["isError"] is False
        assert "installed" in resp["result"]["content"][0]["text"]
        mock_install.assert_called_once_with("coder", True)

    def test_install_plugin_with_deps_installed_ok_is_success(self):
        from localm.plugins.deps import InstallResult
        server, _ = _server()
        ok_result = InstallResult(ok=True, installed=["voice-lib>=1.0"])
        with patch("localm.plugins.engine.PluginManager.set_installed_state"), \
             patch("localm.plugins.engine.PluginManager.plugin_missing_deps", return_value=True), \
             patch("localm.plugins.engine.PluginManager.install_plugin_deps",
                   return_value=ok_result):
            resp = _req(server, "tools/call",
                        {"name": "install_plugin", "arguments": {"plugin": "voice"}})
        assert resp["result"]["isError"] is False
        assert "successfully installed and enabled" in resp["result"]["content"][0]["text"]

    def test_install_plugin_reports_dependency_install_failure(self):
        # The tool must not report a blanket "successfully installed" when the
        # plugin's declared pip extras actually failed to install.
        from localm.plugins.deps import InstallResult
        server, _ = _server()
        fail_result = InstallResult(ok=False, failed=["voice-lib>=1.0"],
                                    error="network down")
        with patch("localm.plugins.engine.PluginManager.set_installed_state"), \
             patch("localm.plugins.engine.PluginManager.plugin_missing_deps", return_value=True), \
             patch("localm.plugins.engine.PluginManager.install_plugin_deps",
                   return_value=fail_result) as mock_deps:
            resp = _req(server, "tools/call",
                        {"name": "install_plugin", "arguments": {"plugin": "voice"}})
        assert resp["result"]["isError"] is True
        text = resp["result"]["content"][0]["text"]
        assert "installed and enabled" in text
        assert "voice-lib>=1.0" in text
        assert "network down" in text
        mock_deps.assert_called_once_with("voice")

    @pytest.mark.parametrize("tool_name,expected_bool,expected_substring", [
        ("enable_plugin", True, "enabled"),
        ("disable_plugin", False, "disabled"),
    ])
    def test_enable_disable_plugin(self, tool_name, expected_bool, expected_substring):
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.set_enabled_state") as mock_set:
            resp = _req(server, "tools/call",
                        {"name": tool_name, "arguments": {"plugin": "coder"}})
        assert resp["result"]["isError"] is False
        assert expected_substring in resp["result"]["content"][0]["text"]
        mock_set.assert_called_once_with("coder", expected_bool)

    def test_uninstall_plugin(self):
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.uninstall") as mock_uninstall:
            resp = _req(server, "tools/call",
                        {"name": "uninstall_plugin", "arguments": {"plugin": "coder", "delete_data": True}})
        assert resp["result"]["isError"] is False
        assert "uninstalled" in resp["result"]["content"][0]["text"]
        mock_uninstall.assert_called_once_with("coder", delete_data=True)

    def test_uninstall_plugin_reports_degraded_removal(self):
        # PluginManager.uninstall() returns False (not an exception) when the
        # installed directory could not actually be removed - a locked file, an
        # AV hold, or a permission denial, all reachable on Windows. A bare
        # "successfully uninstalled" there reports a step that did not complete.
        server, _ = _server()
        with patch("localm.plugins.engine.PluginManager.is_installed", return_value=True), \
             patch("localm.plugins.engine.PluginManager.uninstall", return_value=False) as mock_uninstall:
            resp = _req(server, "tools/call",
                        {"name": "uninstall_plugin", "arguments": {"plugin": "coder"}})
        assert resp["result"]["isError"] is True
        text = resp["result"]["content"][0]["text"]
        assert "could not be fully removed" in text
        assert "successfully uninstalled" not in text
        mock_uninstall.assert_called_once_with("coder", delete_data=False)
