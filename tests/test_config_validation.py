# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config update validation (localm.settings_schema.validate_update) and its two
consumers: the `localm config` CLI command and PATCH /v1/config.

Without the shared validator both call sites write any key of any type with no
checking: `localm config bogus x` persists an unknown key, a scalar clobbers a
list key (plugins_enabled / net_allow), and PATCH {net_deny: null} silently wipes
the SSRF deny-list.
"""

from unittest.mock import patch

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG
from localm.discover import (FREE_SCOPE_DEVICE, FREE_SCOPE_PROCESS, GPU_PROBE_OK,
                             GPU_PROBE_TIMEOUT)


# --------------------------------------------------------------------------- #
#  validate_update - unit
# --------------------------------------------------------------------------- #

class TestValidateUpdate:
    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown config key"):
            ss.validate_update({"definitely_not_a_key": 1})

    def test_number_coerced_from_string_and_range_checked(self):
        assert ss.validate_update({"n_ctx": "8192"}) == {"n_ctx": 8192}
        assert isinstance(ss.validate_update({"n_ctx": "8192"})["n_ctx"], int)
        # temperature has max=2
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"temperature": 5})
        # port has max=65535
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"port": 999999})
        with pytest.raises(ValueError, match="expected an integer"):
            ss.validate_update({"n_ctx": "not-a-number"})

    def test_number_rejects_bool(self):
        with pytest.raises(ValueError, match="not a boolean"):
            ss.validate_update({"n_ctx": True})

    def test_gpu_layers_out_of_range(self):
        # n_gpu_layers has a ceiling, so an absurd value never reaches the native
        # layer.
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"n_gpu_layers": 1111})
        assert ss.validate_update({"n_gpu_layers": 99}) == {"n_gpu_layers": 99}

    def test_n_ctx_out_of_range(self):
        # n_ctx/n_ctx_max/n_ctx_grow have a ceiling as well as a floor. A real
        # long-context value is accepted first, so a too-tight ceiling goes red
        # here.
        assert ss.validate_update({"n_ctx": 131072}) == {"n_ctx": 131072}
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"n_ctx": 10**18})
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"n_ctx_max": 10**18})
        # 0 = unlimited sentinel must still survive a ceiling on the top end.
        assert ss.validate_update({"n_ctx_max": 0}) == {"n_ctx_max": 0}
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"n_ctx_grow": 10**18})
        assert ss.validate_update({"n_ctx_grow": 4096}) == {"n_ctx_grow": 4096}

    @pytest.mark.parametrize("value", ["1", 1])
    def test_main_gpu_index_coerces_to_int(self, value):
        # `localm config main_gpu_index 1` arrives as the string "1" (HIDDEN
        # widgets skip the generic number coercion, so this key is
        # special-cased); PATCH /v1/config from the GUI sends a native JSON
        # int. Both are stored as a real int.
        result = ss.validate_update({"main_gpu_index": value})
        assert result == {"main_gpu_index": 1}
        assert isinstance(result["main_gpu_index"], int)

    def test_main_gpu_index_null_clears_it(self):
        assert ss.validate_update({"main_gpu_index": None}) == {"main_gpu_index": None}

    def test_main_gpu_index_rejects_negative(self):
        with pytest.raises(ValueError, match="below the minimum"):
            ss.validate_update({"main_gpu_index": -1})

    def test_main_gpu_index_rejects_non_integer(self):
        with pytest.raises(ValueError, match="expected an integer"):
            ss.validate_update({"main_gpu_index": "not-a-number"})

    def test_main_gpu_index_rejects_bool(self):
        with pytest.raises(ValueError, match="not a boolean"):
            ss.validate_update({"main_gpu_index": True})

    def test_main_gpu_index_rejects_index_above_sanity_ceiling(self):
        # main_gpu_index has the same ceiling as the sibling gpu_split_indices
        # field, so an out-of-range value never reaches the ctypes.c_int32
        # main_gpu field.
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"main_gpu_index": 500_000})

    def test_main_gpu_index_allows_ceiling_boundary(self):
        assert ss.validate_update({"main_gpu_index": ss.MAX_GPU_SPLIT_INDEX}) == \
            {"main_gpu_index": ss.MAX_GPU_SPLIT_INDEX}

    # --- gpu_split_indices / gpu_split_ratios (HIDDEN list-of-number) ----- #

    def test_gpu_split_indices_coerces_csv_string(self):
        # `localm config gpu_split_indices 0,1` arrives as the CLI CSV string.
        result = ss.validate_update({"gpu_split_indices": "0,1"})
        assert result == {"gpu_split_indices": [0, 1]}
        assert all(isinstance(i, int) for i in result["gpu_split_indices"])

    def test_gpu_split_indices_coerces_json_list(self):
        # PATCH /v1/config from the GUI sends a native JSON list of ints.
        result = ss.validate_update({"gpu_split_indices": [0, 1]})
        assert result == {"gpu_split_indices": [0, 1]}
        assert all(isinstance(i, int) for i in result["gpu_split_indices"])

    def test_gpu_split_indices_null_clears_it(self):
        assert ss.validate_update({"gpu_split_indices": None}) == {
            "gpu_split_indices": None}

    def test_gpu_split_indices_empty_string_clears_it(self):
        assert ss.validate_update({"gpu_split_indices": ""}) == {
            "gpu_split_indices": None}

    def test_gpu_split_indices_empty_list_clears_it(self):
        assert ss.validate_update({"gpu_split_indices": []}) == {
            "gpu_split_indices": None}

    def test_gpu_split_indices_rejects_negative(self):
        with pytest.raises(ValueError, match="below the minimum"):
            ss.validate_update({"gpu_split_indices": [0, -1]})

    def test_gpu_split_indices_rejects_bool(self):
        with pytest.raises(ValueError):
            ss.validate_update({"gpu_split_indices": [True, False]})

    def test_gpu_split_indices_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="expected an integer"):
            ss.validate_update({"gpu_split_indices": ["abc"]})

    def test_gpu_split_indices_rejects_index_above_sanity_ceiling(self):
        # An unbounded index drives an unbounded ctypes allocation in
        # discover.apply_gpu_split (a raw tensor_split buffer sized to the
        # highest configured index), so it is rejected at write time.
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"gpu_split_indices": [0, 500_000]})

    def test_gpu_split_indices_allows_ceiling_boundary(self):
        assert ss.validate_update({"gpu_split_indices": [0, ss.MAX_GPU_SPLIT_INDEX]}) == \
            {"gpu_split_indices": [0, ss.MAX_GPU_SPLIT_INDEX]}

    def test_gpu_split_ratios_coerces_csv_string(self):
        result = ss.validate_update({"gpu_split_ratios": "0.6,0.4"})
        assert result == {"gpu_split_ratios": [0.6, 0.4]}
        assert all(isinstance(r, float) for r in result["gpu_split_ratios"])

    def test_gpu_split_ratios_null_clears_it(self):
        assert ss.validate_update({"gpu_split_ratios": None}) == {
            "gpu_split_ratios": None}

    def test_gpu_split_ratios_rejects_zero_or_negative(self):
        with pytest.raises(ValueError, match="greater than 0"):
            ss.validate_update({"gpu_split_ratios": [0.6, 0]})
        with pytest.raises(ValueError, match="greater than 0"):
            ss.validate_update({"gpu_split_ratios": [-0.1, 0.5]})

    def test_gpu_split_ratios_rejects_non_finite(self):
        # A non-finite ratio is rejected before it reaches the native
        # tensor_split ctypes buffer (discover.apply_gpu_split).
        with pytest.raises(ValueError, match="finite"):
            ss.validate_update({"gpu_split_ratios": [1.0, float("inf")]})
        with pytest.raises(ValueError, match="finite"):
            ss.validate_update({"gpu_split_ratios": [1.0, float("nan")]})

    def test_toggle_coercion(self):
        assert ss.validate_update({"require_auth": "true"}) == {"require_auth": True}
        assert ss.validate_update({"require_auth": "false"}) == {"require_auth": False}
        assert ss.validate_update({"require_auth": True}) == {"require_auth": True}
        with pytest.raises(ValueError, match="expected a boolean"):
            ss.validate_update({"require_auth": "maybe"})

    def test_select_must_be_in_options(self):
        assert ss.validate_update({"mode": "log"}) == {"mode": "log"}
        with pytest.raises(ValueError, match="is not one of"):
            ss.validate_update({"mode": "bogus"})
        # nullable SELECT (chat_mode default None): "" / None -> None (inherit)
        assert ss.validate_update({"chat_mode": ""}) == {"chat_mode": None}
        assert ss.validate_update({"chat_mode": None}) == {"chat_mode": None}

    def test_list_key_rejects_scalar_clobber(self):
        # a bare string becomes a one-element list, not a scalar overwrite
        assert ss.validate_update({"net_allow": "x.com"}) == {"net_allow": ["x.com"]}
        assert ss.validate_update({"net_deny": ["a.com", "b.com"]}) == {
            "net_deny": ["a.com", "b.com"]}
        assert ss.validate_update({"net_allow": "a.com, b.com"}) == {
            "net_allow": ["a.com", "b.com"]}

    def test_list_key_null_is_rejected_not_wiped(self):
        # net_deny:null must NOT silently become null (SSRF block wipe)
        with pytest.raises(ValueError, match="a value is required"):
            ss.validate_update({"net_deny": None})

    def test_nullable_text_blank_becomes_none(self):
        assert ss.validate_update({"comfy_output_dir": ""}) == {"comfy_output_dir": None}

    def test_cors_origins_forms(self):
        assert ss.validate_update({"cors_origins": ""}) == {"cors_origins": None}
        assert ss.validate_update({"cors_origins": "*"}) == {"cors_origins": "*"}
        assert ss.validate_update({"cors_origins": "https://a, https://b"}) == {
            "cors_origins": ["https://a", "https://b"]}
        assert ss.validate_update({"cors_origins": ["*"]}) == {"cors_origins": ["*"]}

    def test_cors_origins_rejects_non_string_members(self):
        with pytest.raises(ValueError, match="expected a list of strings"):
            ss.validate_update({"cors_origins": [1, None, {}]})

    def test_cors_origins_rejects_too_many(self):
        with pytest.raises(ValueError, match="too many"):
            ss.validate_update(
                {"cors_origins": ["http://x"] * (ss.MAX_CORS_ORIGINS + 1)})

    def test_cors_origins_allows_cap_boundary(self):
        origins = ["http://x"] * ss.MAX_CORS_ORIGINS
        assert ss.validate_update({"cors_origins": origins}) == {
            "cors_origins": origins}

    def test_gpu_split_neighbor_contract_unaffected_by_cors_fix(self):
        # These two callers depend on the shared _to_str_list helper's
        # str()-coercion of a native JSON list of numbers before _to_number
        # parses each token.
        assert ss.validate_update({"gpu_split_indices": [0, 1]}) == {
            "gpu_split_indices": [0, 1]}
        result = ss.validate_update({"gpu_split_ratios": [0.6, 0.4]})
        assert result == {"gpu_split_ratios": [0.6, 0.4]}
        assert all(isinstance(r, float) for r in result["gpu_split_ratios"])

    def test_plugins_dict_roundtrips_scalar_rejected(self):
        assert ss.validate_update({"plugins": {"image": {"x": 1}}}) == {
            "plugins": {"image": {"x": 1}}}
        with pytest.raises(ValueError, match="expected an object"):
            ss.validate_update({"plugins": "nope"})

    def test_plugins_enabled_list_required(self):
        assert ss.validate_update({"plugins_enabled": ["chat"]}) == {
            "plugins_enabled": ["chat"]}
        with pytest.raises(ValueError, match="expected a list"):
            ss.validate_update({"plugins_enabled": "chat"})

    def test_coder_index_timeout_settable(self):
        # coder_index_timeout is a real schema-backed field, so `localm config
        # coder_index_timeout 30` works.
        assert ss.validate_update({"coder_index_timeout": "30"}) == {
            "coder_index_timeout": 30}
        with pytest.raises(ValueError, match="below the minimum"):
            ss.validate_update({"coder_index_timeout": -1})

    # --- PATHLIST (rag_allowed_roots): coerce, resolve, confine ----------- #

    def test_pathlist_coerces_list_and_resolves(self, tmp_path):
        from pathlib import Path
        a = tmp_path / "docs"
        a.mkdir()
        out = ss.validate_update({"rag_allowed_roots": [str(a)]})
        assert [Path(p) for p in out["rag_allowed_roots"]] == [a.resolve()]

    def test_pathlist_accepts_comma_string_and_dedups(self, tmp_path):
        a = tmp_path / "docs"
        a.mkdir()
        out = ss.validate_update({"rag_allowed_roots": f"{a}, {a}"})
        assert len(out["rag_allowed_roots"]) == 1, "duplicate roots collapse to one"

    def test_pathlist_rejects_credential_dir(self, tmp_path):
        # A folder whose path contains a credential dir name (.ssh) can never be
        # indexed, so it is refused at save time with a clear error rather than
        # stored and ignored later.
        bad = tmp_path / ".ssh" / "keys"
        with pytest.raises(ValueError, match="credential"):
            ss.validate_update({"rag_allowed_roots": [str(bad)]})

    def test_pathlist_null_rejected_not_wiped(self):
        # Like the LIST keys: null must not silently blank the list.
        with pytest.raises(ValueError, match="a value is required"):
            ss.validate_update({"rag_allowed_roots": None})

    def test_indexing_mode_select_validated(self):
        assert ss.validate_update({"rag_indexing_mode": "blacklist"}) == {
            "rag_indexing_mode": "blacklist"}
        assert ss.validate_update({"rag_indexing_mode": "whitelist"}) == {
            "rag_indexing_mode": "whitelist"}
        with pytest.raises(ValueError, match="is not one of"):
            ss.validate_update({"rag_indexing_mode": "everything"})


def test_plugins_key_is_in_default_config_and_schema():
    # the per-plugin config namespace
    assert DEFAULT_CONFIG.get("plugins") == {}
    field_keys = {f.key for f in ss.CORE_FIELDS}
    assert "plugins" in field_keys


# --------------------------------------------------------------------------- #
#  `localm config` CLI - via the CliRunner harness
# --------------------------------------------------------------------------- #

class TestConfigCli:
    def _cfg(self):
        from localm.config import load_config
        return load_config()

    def test_valid_int_persists_as_int(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "n_ctx", "8192"])
        assert r.exit_code == 0, r.output
        assert self._cfg()["n_ctx"] == 8192

    def test_unknown_key_rejected(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "bogus_key", "1"])
        assert r.exit_code != 0
        assert "unknown config key" in r.output.lower()
        assert "bogus_key" not in self._cfg()

    def test_out_of_range_rejected(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "temperature", "9"])
        assert r.exit_code != 0
        assert "maximum" in r.output.lower()

    def test_scalar_cannot_clobber_list_key(self, cli_runner):
        from localm.cli import main
        # plugins_enabled is a HIDDEN list managed by the engine: a scalar string
        # must not overwrite it with a bare string
        r = cli_runner.invoke(main, ["config", "plugins_enabled", "chat"])
        assert r.exit_code != 0
        assert self._cfg().get("plugins_enabled", []) == []

    def test_bad_enum_rejected(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "mode", "loud"])
        assert r.exit_code != 0
        assert "mode" in self._cfg()  # default untouched
        assert self._cfg()["mode"] == DEFAULT_CONFIG["mode"]

    def test_main_gpu_index_settable(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "main_gpu_index", "1"])
        assert r.exit_code == 0, r.output
        assert self._cfg()["main_gpu_index"] == 1

    def test_main_gpu_index_rejects_negative(self, cli_runner):
        from localm.cli import main
        # "--" so click's parser treats "-1" as the VALUE argument, not a
        # (nonexistent) option flag.
        r = cli_runner.invoke(main, ["config", "main_gpu_index", "--", "-1"])
        assert r.exit_code != 0
        assert self._cfg().get("main_gpu_index") is None   # default untouched


# --------------------------------------------------------------------------- #
#  `localm config gpu_split_indices` / `gpu_split_ratios` CLI                 #
# --------------------------------------------------------------------------- #

class TestGpuSplitCli:
    def _cfg(self):
        from localm.config import load_config
        return load_config()

    def test_gpu_split_indices_settable(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "gpu_split_indices", "0,1"])
        assert r.exit_code == 0, r.output
        assert self._cfg()["gpu_split_indices"] == [0, 1]

    def test_gpu_split_indices_rejects_negative(self, cli_runner):
        from localm.cli import main
        # "--" so click's parser treats "-1" as the VALUE argument, not a
        # (nonexistent) option flag.
        r = cli_runner.invoke(main, ["config", "gpu_split_indices", "--", "0,-1"])
        assert r.exit_code != 0
        assert self._cfg().get("gpu_split_indices") is None   # default untouched

    def test_gpu_split_ratios_settable(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "gpu_split_ratios", "0.6,0.4"])
        assert r.exit_code == 0, r.output
        assert self._cfg()["gpu_split_ratios"] == [0.6, 0.4]

    def test_gpu_split_ratios_rejects_zero(self, cli_runner):
        from localm.cli import main
        r = cli_runner.invoke(main, ["config", "gpu_split_ratios", "0.6,0"])
        assert r.exit_code != 0
        assert "greater than 0" in r.output
        assert self._cfg().get("gpu_split_ratios") is None   # default untouched


# --------------------------------------------------------------------------- #
#  `localm gpus` CLI                                                          #
# --------------------------------------------------------------------------- #

class TestGpusCli:
    _GPUS = [
        {"index": 0, "name": "RTX 4090", "total": 24 * 1024 ** 3, "free": 20 * 1024 ** 3},
        {"index": 1, "name": "RTX 3060", "total": 12 * 1024 ** 3, "free": 10 * 1024 ** 3},
    ]

    def test_lists_detected_gpus(self, cli_runner):
        from localm.cli import main
        with patch("localm.discover.list_gpus",
                   return_value=(self._GPUS, GPU_PROBE_OK)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "RTX 4090" in r.output
        assert "RTX 3060" in r.output
        assert "24.0 GB total" in r.output

    def test_marks_the_configured_device(self, cli_runner):
        from localm.cli import main
        cli_runner.invoke(main, ["config", "main_gpu_index", "1"])
        with patch("localm.discover.list_gpus",
                   return_value=(self._GPUS, GPU_PROBE_OK)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "configured" in r.output.lower()

    def test_no_gpus_detected(self, cli_runner):
        from localm.cli import main
        # A COMPLETED probe that finds nothing (status OK): the genuine
        # no-torch/no-nvidia-smi box, which keeps the original message.
        with patch("localm.discover.list_gpus", return_value=([], GPU_PROBE_OK)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "no gpus detected" in r.output.lower()

    def test_warns_when_configured_index_is_stale(self, cli_runner):
        from localm.cli import main
        cli_runner.invoke(main, ["config", "main_gpu_index", "5"])
        with patch("localm.discover.list_gpus",
                   return_value=(self._GPUS[:1], GPU_PROBE_OK)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "does not match any gpu" in r.output.lower()

    def test_timeout_reports_retry_not_no_torch(self, cli_runner, monkeypatch):
        """A GPU probe that overruns the deadline (a cold ROCm/CUDA driver init)
        must be reported as a TIMEOUT with a retry hint, NOT as 'no torch / no
        GPU'. torch IS installed and a warm retry works, so 'no torch' is a false
        claim."""
        import threading
        from localm.cli import main

        release = threading.Event()

        def _slow():
            release.wait(10)     # a cold driver init that overruns the deadline
            return [{"index": 0, "name": "GPU0", "total": 8, "free": 8}]

        monkeypatch.setattr("localm.discover._list_gpus_probe", _slow)
        # Shrink the CLI's generous deadline so the test is fast; the probe still
        # overruns it, exercising the timeout branch with the real status logic.
        monkeypatch.setattr("localm.discover._GPU_PROBE_CLI_DEADLINE", 0.2)
        r = cli_runner.invoke(main, ["gpus"])
        release.set()            # let the abandoned probe thread finish now
        assert r.exit_code == 0, r.output
        out = r.output.lower()
        assert "no torch" not in out, (
            "a timed-out GPU probe was misreported as 'no torch' (rule 5)")
        assert ("timed out" in out or "timeout" in out)
        assert ("again" in out or "retry" in out)

    @staticmethod
    def _flat(output):
        # Rich's console wraps long lines to the render width, so collapse all
        # whitespace before the substring check.
        return " ".join(output.split())

    # The `gpus` command omits the free VRAM figure entirely and says so. These
    # cover both dimensions: device/process free_scope, and fresh/stale probe
    # status.
    def test_process_scoped_free_is_omitted(self, cli_runner):
        from localm.cli import main
        gpus = [{**self._GPUS[0], "free_scope": FREE_SCOPE_PROCESS}]
        with patch("localm.discover.list_gpus", return_value=(gpus, GPU_PROBE_OK)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "GB free" not in r.output
        assert "24.0 GB total" in r.output
        assert "free VRAM reading unavailable on this platform" in self._flat(r.output)

    def test_device_scoped_fresh_free_is_shown(self, cli_runner):
        from localm.cli import main
        gpus = [{**self._GPUS[0], "free_scope": FREE_SCOPE_DEVICE}]
        with patch("localm.discover.list_gpus", return_value=(gpus, GPU_PROBE_OK)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "20.0 GB free" in r.output
        assert "free VRAM reading unavailable" not in self._flat(r.output)

    def test_stale_probe_status_omits_free_even_with_device_scope(self, cli_runner):
        """A served last-known-good list (TIMEOUT/BUSY/INCONCLUSIVE) is not a
        current measurement even when it carries a FREE_SCOPE_DEVICE tag from
        the earlier successful probe that produced it - list_gpus()'s own
        docstring documents exactly this. Freshness AND scope must both hold."""
        from localm.cli import main
        gpus = [{**self._GPUS[0], "free_scope": FREE_SCOPE_DEVICE}]
        with patch("localm.discover.list_gpus",
                   return_value=(gpus, GPU_PROBE_TIMEOUT)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "GB free" not in r.output
        assert "24.0 GB total" in r.output
        assert "free VRAM reading unavailable on this platform" in self._flat(r.output)

    def test_untagged_free_scope_defers_to_raw_reading_check(self, cli_runner):
        """No free_scope key at all (never happens from the real discover.list_gpus,
        which always tags it, but is what a raw/legacy entry would look like) must
        fall back to gpu_usage.raw_reading_is_process_scoped() exactly like
        doctor.py's torch VRAM check does, not silently assume either scope."""
        from localm.cli import main
        gpus = [dict(self._GPUS[0])]   # no "free_scope" key
        with patch("localm.discover.list_gpus", return_value=(gpus, GPU_PROBE_OK)), \
             patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=True):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "GB free" not in r.output
        assert "free VRAM reading unavailable on this platform" in self._flat(r.output)

        with patch("localm.discover.list_gpus", return_value=(gpus, GPU_PROBE_OK)), \
             patch("localm.gpu_usage.raw_reading_is_process_scoped", return_value=False):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "20.0 GB free" in r.output
        assert "free VRAM reading unavailable" not in self._flat(r.output)

    def test_no_free_reading_has_no_caveat(self, cli_runner):
        """free itself is None (no reading at all) - nothing to caveat, even when
        free_scope claims PROCESS scope, since no free figure is ever printed."""
        from localm.cli import main
        gpus = [{**self._GPUS[0], "free": None, "free_scope": FREE_SCOPE_PROCESS}]
        with patch("localm.discover.list_gpus", return_value=(gpus, GPU_PROBE_OK)):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "GB free" not in r.output
        assert "free VRAM reading unavailable" not in self._flat(r.output)


# --------------------------------------------------------------------------- #
#  PATCH /v1/config - via TestClient
# --------------------------------------------------------------------------- #

class TestPatchConfig:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from localm.inference.http_server import create_app
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        # Management routes need the loopback shell token in open mode; the
        # GUI carries it, so do these config-validation tests.
        app = create_app(None)
        return TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"})

    def test_unknown_key_400(self, client):
        r = client.patch("/v1/config", json={"hax": 1})
        assert r.status_code == 400

    def test_net_deny_null_rejected_not_wiped(self, client):
        # clobbering net_deny to null removes SSRF blocks - must 400
        r = client.patch("/v1/config", json={"net_deny": None})
        assert r.status_code == 400
        # and net_deny stays its list default
        got = client.get("/v1/config").json()
        assert got["net_deny"] == []

    def test_wrong_type_400(self, client):
        assert client.patch("/v1/config", json={"n_ctx": "abc"}).status_code == 400
        assert client.patch("/v1/config", json={"port": 999999}).status_code == 400

    def test_valid_patch_persists(self, client):
        r = client.patch("/v1/config", json={"n_ctx": 8192, "mode": "log"})
        assert r.status_code == 200
        got = client.get("/v1/config").json()
        assert got["n_ctx"] == 8192 and got["mode"] == "log"

    def test_plugins_dict_accepted(self, client):
        # per-plugin config round-trips
        r = client.patch("/v1/config", json={"plugins": {"image": {"comfy": {"output_dir": "x"}}}})
        assert r.status_code == 200
        got = client.get("/v1/config").json()
        assert got["plugins"]["image"]["comfy"]["output_dir"] == "x"

    def test_string_number_coerced(self, client):
        # the flat GUI form submits strings; the server coerces them
        r = client.patch("/v1/config", json={"n_gpu_layers": "20"})
        assert r.status_code == 200
        assert client.get("/v1/config").json()["n_gpu_layers"] == 20

    def test_gpu_layers_out_of_range_400(self, client):
        r = client.patch("/v1/config", json={"n_gpu_layers": 1111})
        assert r.status_code == 400
        assert "maximum" in r.text

    def test_chat_background_round_trips(self, client):
        uri = "data:image/jpeg;base64,iVBORw0KGgo="
        r = client.patch("/v1/config", json={"chat_background": uri})
        assert r.status_code == 200
        assert client.get("/v1/config").json()["chat_background"] == uri

    def test_chat_background_rejects_a_url_400(self, client):
        r = client.patch("/v1/config", json={"chat_background": "http://evil.example/x.jpg"})
        assert r.status_code == 400
        assert client.get("/v1/config").json()["chat_background"] == ""

    def test_user_name_round_trips_and_strips(self, client):
        r = client.patch("/v1/config", json={"user_name": "  Matt  "})
        assert r.status_code == 200
        assert client.get("/v1/config").json()["user_name"] == "Matt"


# --------------------------------------------------------------------------- #
#  /v1/config instance_id: a stable per-data-directory id, so the GUI can tell
#  a restart of THIS install apart from a different install sharing the browser
#  origin, and never renders a foreign install's cached conversations.
# --------------------------------------------------------------------------- #

class TestConfigInstanceId:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from localm.inference.http_server import create_app
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        app = create_app(None)
        return TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"})

    def test_instance_id_present_and_stable(self, client, tmp_path):
        first = client.get("/v1/config").json()["instance_id"]
        assert first, "instance_id must be a non-empty string"
        second = client.get("/v1/config").json()["instance_id"]
        assert second == first, "the id must not change across requests"
        # Persisted under the data dir, not re-minted on the next read.
        marker = tmp_path / ".localm" / "instance_id.txt"
        assert marker.is_file()
        assert marker.read_text(encoding="utf-8").strip() == first

    def test_instance_id_differs_per_data_dir(self, tmp_path, monkeypatch):
        # Two DIFFERENT data directories must never mint the same id - this is
        # exactly what lets the GUI tell two installs apart.
        from fastapi.testclient import TestClient
        from localm.inference.http_server import create_app
        import localm.config as cfg
        home_a = tmp_path / "a"
        home_a.mkdir()
        home_b = tmp_path / "b"
        home_b.mkdir()

        monkeypatch.setattr(cfg, "HOME_DIR", home_a)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home_a / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home_a / "registry.json")
        app_a = create_app(None)
        client_a = TestClient(
            app_a, headers={"Authorization": f"Bearer {app_a.state.shell_token}"})
        id_a = client_a.get("/v1/config").json()["instance_id"]

        monkeypatch.setattr(cfg, "HOME_DIR", home_b)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home_b / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home_b / "registry.json")
        app_b = create_app(None)
        client_b = TestClient(
            app_b, headers={"Authorization": f"Bearer {app_b.state.shell_token}"})
        id_b = client_b.get("/v1/config").json()["instance_id"]

        assert id_a != id_b

    def test_instance_id_is_readonly_on_patch(self, client):
        # Echoing the whole GET response back through PATCH (as the settings
        # form does) must not be rejected for the server-injected instance_id
        # field, and PATCH must never be able to overwrite it.
        got = client.get("/v1/config").json()
        r = client.patch("/v1/config", json=got)
        assert r.status_code == 200
        got2 = client.get("/v1/config").json()
        assert got2["instance_id"] == got["instance_id"]


# --------------------------------------------------------------------------- #
#  /v1/config instance_port: the live bound port, which can differ from the
#  persisted "port" default (an explicit -p override, or an auto-bump onto a
#  different free port, is never written back to disk).
# --------------------------------------------------------------------------- #

class TestConfigInstancePort:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from localm.inference.http_server import create_app
        import localm.config as cfg
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.setattr(cfg, "HOME_DIR", home)
        monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
        app = create_app(None)
        return TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"})

    def test_instance_port_reflects_app_state(self, client):
        # A bare create_app() (no instances.advertise() context, as in every
        # test client here) never sets instance_port - confirm the absence is
        # reported as None rather than raising.
        assert client.get("/v1/config").json()["instance_port"] is None
        # Once a surface advertises a live port (instances.advertise(), or an
        # explicit -p override that differs from the persisted default), the
        # route must reflect exactly that value, not the persisted "port" key.
        client.app.state.instance_port = 1111
        got = client.get("/v1/config").json()
        assert got["instance_port"] == 1111
        assert got["port"] != 1111, "persisted default is the untouched 8642, not the live port"

    def test_instance_port_is_readonly_on_patch(self, client):
        # Same contract as instance_id above: echoing the whole GET response
        # back through PATCH must not be rejected for instance_port, and PATCH
        # must never be able to overwrite it (the server bind is not a setting).
        client.app.state.instance_port = 1111
        got = client.get("/v1/config").json()
        r = client.patch("/v1/config", json=got)
        assert r.status_code == 200
        assert client.get("/v1/config").json()["instance_port"] == 1111


class TestInstanceIdUnit:
    """Direct unit coverage of config.instance_id(), independent of the HTTP layer."""

    def test_mints_and_persists(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        val = cfg.instance_id()
        assert val
        marker = tmp_path / "instance_id.txt"
        assert marker.is_file()
        assert marker.read_text(encoding="utf-8").strip() == val

    def test_reuses_existing_across_calls(self, tmp_path, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        first = cfg.instance_id()
        second = cfg.instance_id()
        assert first == second, "never regenerated on a normal restart"

    def test_recovers_from_empty_marker_file(self, tmp_path, monkeypatch):
        # A zero-byte / corrupt marker (a truncated write) does not crash the
        # caller: a fresh id is minted and persisted.
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        marker = tmp_path / "instance_id.txt"
        tmp_path.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
        val = cfg.instance_id()
        assert val
        assert marker.read_text(encoding="utf-8").strip() == val


class TestGpuLayersCliRange:
    def test_run_rejects_out_of_range_gpu_layers(self, cli_runner):
        from localm.cli import main
        result = cli_runner.invoke(main, ["run", "-g", "1111", "dummy-model"])
        assert result.exit_code != 0
        # The IntRange ceiling (1000) appears in the error message.
        assert "1000" in result.output
