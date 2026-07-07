# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config update validation (localm.settings_schema.validate_update) and its two
consumers: the `localm config` CLI command and PATCH /v1/config.

Before the shared validator, both call sites wrote any key/any type with no
checking: `localm config bogus x` persisted an unknown key, a scalar clobbered a
list key (plugins_enabled / net_allow), and PATCH {net_deny: null} silently wiped
the SSRF deny-list (P0-6 / SEC-2 / BUG-3).
"""

from unittest.mock import patch

import pytest

from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG


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
        # n_gpu_layers had min=0 but no max, so an absurd value (1111) sailed
        # through to the native layer and surfaced as a misleading "repair
        # llama" error. It now has a ceiling.
        with pytest.raises(ValueError, match="above the maximum"):
            ss.validate_update({"n_gpu_layers": 1111})
        assert ss.validate_update({"n_gpu_layers": 99}) == {"n_gpu_layers": 99}

    def test_main_gpu_index_coerces_cli_string_to_int(self):
        # `localm config main_gpu_index 1` arrives as the string "1"; it must
        # be stored as a real int, not left stringly-typed (HIDDEN widgets
        # skip the generic number coercion, so this key is special-cased).
        assert ss.validate_update({"main_gpu_index": "1"}) == {"main_gpu_index": 1}
        assert isinstance(ss.validate_update({"main_gpu_index": "1"})["main_gpu_index"], int)

    def test_main_gpu_index_accepts_json_int(self):
        # PATCH /v1/config from the GUI sends a native JSON int.
        assert ss.validate_update({"main_gpu_index": 1}) == {"main_gpu_index": 1}

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
        # SEC-2: net_deny:null must NOT silently become null (SSRF block wipe)
        with pytest.raises(ValueError, match="a value is required"):
            ss.validate_update({"net_deny": None})

    def test_nullable_text_blank_becomes_none(self):
        assert ss.validate_update({"comfy_output_dir": ""}) == {"comfy_output_dir": None}

    def test_cors_origins_forms(self):
        assert ss.validate_update({"cors_origins": ""}) == {"cors_origins": None}
        assert ss.validate_update({"cors_origins": "*"}) == {"cors_origins": "*"}
        assert ss.validate_update({"cors_origins": "https://a, https://b"}) == {
            "cors_origins": ["https://a", "https://b"]}

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
        # Previously read directly off raw config (checkpoint._index_deadline)
        # but never registered in DEFAULT_CONFIG/CORE_FIELDS, so `localm config
        # coder_index_timeout 30` raised "unknown config key" - the only way to
        # set it was hand-editing config.json. Now a real, schema-backed field.
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
        # indexed, so it is refused at save time with a clear error (not silently
        # stored then ignored at index time).
        bad = tmp_path / ".ssh" / "keys"
        with pytest.raises(ValueError, match="credential"):
            ss.validate_update({"rag_allowed_roots": [str(bad)]})

    def test_pathlist_null_rejected_not_wiped(self):
        # Like the LIST keys (SEC-2): null must not silently blank the list.
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
    # FAC-1: per-plugin config namespace now has a documented home
    assert DEFAULT_CONFIG.get("plugins") == {}
    field_keys = {f.key for f in ss.CORE_FIELDS}
    assert "plugins" in field_keys


# --------------------------------------------------------------------------- #
#  `localm config` CLI (BUG-3) - via the CliRunner harness
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
#  `localm gpus` CLI                                                          #
# --------------------------------------------------------------------------- #

class TestGpusCli:
    _GPUS = [
        {"index": 0, "name": "RTX 4090", "total": 24 * 1024 ** 3, "free": 20 * 1024 ** 3},
        {"index": 1, "name": "RTX 3060", "total": 12 * 1024 ** 3, "free": 10 * 1024 ** 3},
    ]

    def test_lists_detected_gpus(self, cli_runner):
        from localm.cli import main
        with patch("localm.discover.list_gpus", return_value=self._GPUS):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "RTX 4090" in r.output
        assert "RTX 3060" in r.output
        assert "24.0 GB total" in r.output

    def test_marks_the_configured_device(self, cli_runner):
        from localm.cli import main
        cli_runner.invoke(main, ["config", "main_gpu_index", "1"])
        with patch("localm.discover.list_gpus", return_value=self._GPUS):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "configured" in r.output.lower()

    def test_no_gpus_detected(self, cli_runner):
        from localm.cli import main
        with patch("localm.discover.list_gpus", return_value=[]):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "no gpus detected" in r.output.lower()

    def test_warns_when_configured_index_is_stale(self, cli_runner):
        from localm.cli import main
        cli_runner.invoke(main, ["config", "main_gpu_index", "5"])
        with patch("localm.discover.list_gpus", return_value=self._GPUS[:1]):
            r = cli_runner.invoke(main, ["gpus"])
        assert r.exit_code == 0, r.output
        assert "does not match any gpu" in r.output.lower()


# --------------------------------------------------------------------------- #
#  PATCH /v1/config (SEC-2 / FAC-1) - via TestClient
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
        # H5: management routes need the loopback shell token in open mode; the
        # GUI carries it, so do these config-validation tests.
        app = create_app(None)
        return TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"})

    def test_unknown_key_400(self, client):
        r = client.patch("/v1/config", json={"hax": 1})
        assert r.status_code == 400

    def test_net_deny_null_rejected_not_wiped(self, client):
        # SEC-2: clobbering net_deny to null removes SSRF blocks - must 400
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
        # FAC-1: per-plugin config round-trips (was 400 "Unknown config keys: plugins")
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


# --------------------------------------------------------------------------- #
#  /v1/config instance_id (AUD-INSTANCEID)                                    #
#                                                                              #
#  A stable per-data-directory id, so the GUI can tell a normal restart of    #
#  THIS install apart from a different install that happens to share the     #
#  browser origin (localStorage is scoped by origin, not by data directory)  #
#  and never render/upload a foreign install's cached conversations.         #
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
        # A zero-byte / corrupt marker (a truncated write) must not crash the
        # caller - a fresh id is minted and persisted (rule 5: do not hide the
        # corruption path behind a crash, but also do not brick the server).
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
        # The IntRange ceiling (1000) appears in the error only post-fix; pre-fix
        # the bare type=int accepted 1111 and failed later on model resolution.
        assert "1000" in result.output
