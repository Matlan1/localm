# SPDX-License-Identifier: AGPL-3.0-or-later
"""Config update validation (localm.settings_schema.validate_update) and its two
consumers: the `localm config` CLI command and PATCH /v1/config.

Before the shared validator, both call sites wrote any key/any type with no
checking: `localm config bogus x` persisted an unknown key, a scalar clobbered a
list key (plugins_enabled / net_allow), and PATCH {net_deny: null} silently wiped
the SSRF deny-list (P0-6 / SEC-2 / BUG-3).
"""

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


class TestGpuLayersCliRange:
    def test_run_rejects_out_of_range_gpu_layers(self, cli_runner):
        from localm.cli import main
        result = cli_runner.invoke(main, ["run", "-g", "1111", "dummy-model"])
        assert result.exit_code != 0
        # The IntRange ceiling (1000) appears in the error only post-fix; pre-fix
        # the bare type=int accepted 1111 and failed later on model resolution.
        assert "1000" in result.output
