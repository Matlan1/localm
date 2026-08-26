# SPDX-License-Identifier: AGPL-3.0-or-later
"""A config.json that parses as valid JSON but is NOT an object (a list, a bare
string, a number, null) resolves to defaults, discarding the user's saved
settings. A genuinely unparseable file warns via _read_json; the
valid-JSON-wrong-shape case would otherwise be silent.

`_merge_stored_config` surfaces that discard once per process, while a MISSING
file stays the benign silent default."""

import pytest

import localm.config as cfg


@pytest.fixture()
def cfg_home(tmp_path, monkeypatch):
    """Point config's frozen CONFIG_FILE/REGISTRY_FILE at a throwaway home."""
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


@pytest.mark.parametrize("bad", ["[1, 2, 3]", '"hello"', "42", "null"])
def test_load_config_non_dict_warns_not_silent(cfg_home, capsys, bad):
    """A present-but-non-object config.json must not silently become defaults -
    the user's settings are being ignored, which has to be surfaced."""
    (cfg_home / "config.json").write_text(bad, encoding="utf-8")
    cfg._warned_bad_config.clear()
    conf = cfg.load_config()
    assert conf["port"] == cfg.DEFAULT_CONFIG["port"]  # falls back to defaults
    err = capsys.readouterr().err.lower()
    assert "config.json" in err and ("not a json object" in err or "ignoring" in err), (
        "a present-but-non-dict config.json must be surfaced, not silently dropped")


def test_load_config_missing_is_silent(cfg_home, capsys):
    """A MISSING config.json is the benign default path - no warning."""
    cfg._warned_bad_config.clear()
    assert not (cfg_home / "config.json").exists()
    cfg.load_config()
    err = capsys.readouterr().err.lower()
    assert "config.json" not in err, "a missing config is a normal default, not a warning"


def test_load_config_non_dict_leaves_file_for_recovery(cfg_home):
    """A pure load must not rewrite the file - it stays hand-recoverable."""
    p = cfg_home / "config.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    before = p.read_bytes()
    cfg.load_config()
    assert p.read_bytes() == before, "load_config must not rewrite a non-dict config.json"


def test_update_config_non_dict_does_not_crash_and_persists(cfg_home):
    """update_config over a non-dict config.json falls back to defaults, applies
    the mutation, and writes a valid config - the bad file is snapshotted to .bak
    by the atomic write, so it stays recoverable."""
    p = cfg_home / "config.json"
    p.write_text('"not an object"', encoding="utf-8")
    cfg._warned_bad_config.clear()
    out = cfg.update_config(lambda c: c.__setitem__("port", 9191))
    assert out["port"] == 9191
    import json
    assert json.loads(p.read_text())["port"] == 9191
    assert (cfg_home / "config.json.bak").exists()  # old (bad) file preserved


class TestUnreadableConfigIsRefusedNotOverwritten:
    """An UNREADABLE config/registry is not the same as an absent one.

    `_read_json` returns the caller's default for both, which is right for the
    read-only consumers (auth, netpolicy, netname, updater all fail safe on
    defaults). It is wrong for a read-modify-write: update_config would merge
    that default, and `_user_delta` would then persist ONLY the key just set,
    replacing every setting the user had while the caller reported success.

    These assert on the FILE before the exception, and catch by hand rather
    than with `pytest.raises`: as a context manager `pytest.raises` fails at the
    end of its `with` block, so a regression reports "DID NOT RAISE
    ConfigUnreadable" and never reaches the file check.

    Distinct from the valid-JSON-wrong-shape case above, which is still
    tolerated.
    """

    def test_update_config_refuses_and_leaves_the_file_alone(self, cfg_home):
        p = cfg_home / "config.json"
        cfg.update_config(lambda c: c.update({
            "net_mode": "off", "llama_runtime_pin": "b1288"}))
        p.write_text("{ this is not json", encoding="utf-8")
        (cfg_home / "config.json.bak").write_text("{ nor this", encoding="utf-8")
        corrupt = p.read_bytes()

        raised = None
        try:
            cfg.update_config(lambda c: c.__setitem__("embedding_model", "x"))
        except cfg.ConfigUnreadable as e:
            raised = e

        assert p.read_bytes() == corrupt, (
            "an unreadable config.json was OVERWRITTEN; every user setting "
            "(including net_mode and llama_runtime_pin) would be gone")
        assert raised is not None, "update_config did not refuse"
        # Names the file, never the path: this message can reach an HTTP error
        # body via inference/routes/config.py.
        assert "config.json" in str(raised)
        assert str(cfg_home) not in str(raised)

    def test_update_registry_refuses_and_leaves_the_file_alone(self, cfg_home):
        """Worse than config: update_registry writes the WHOLE dict, so one
        registration over an unreadable registry leaves only that model."""
        p = cfg_home / "registry.json"
        cfg.update_registry(lambda r: r.__setitem__("a", {"path": "a.gguf"}))
        p.write_text("{ not json", encoding="utf-8")
        corrupt = p.read_bytes()

        raised = None
        try:
            cfg.update_registry(lambda r: r.__setitem__("b", {"path": "b.gguf"}))
        except cfg.ConfigUnreadable as e:
            raised = e

        assert p.read_bytes() == corrupt, (
            "an unreadable registry.json was OVERWRITTEN; every registered "
            "model but the one just added would be gone")
        assert raised is not None, "update_registry did not refuse"

    def test_absent_config_still_writes(self, cfg_home):
        """The control: first run must be unaffected, or the refusal would be
        unfalsifiable (a fix that refused everything would pass the tests above)."""
        p = cfg_home / "config.json"
        assert not p.is_file()
        cfg.update_config(lambda c: c.__setitem__("port", 9191))
        import json
        assert json.loads(p.read_text(encoding="utf-8"))["port"] == 9191

    def test_recovered_from_bak_still_writes(self, cfg_home):
        """The second control: an unreadable PRIMARY whose .bak reads fine is a
        SUCCESSFUL read of real data, so it must not refuse."""
        import json
        p = cfg_home / "config.json"
        p.write_text(json.dumps({"net_mode": "off"}), encoding="utf-8")
        cfg.update_config(lambda c: c.__setitem__("port", 9393))   # rotates .bak
        p.write_text("{ corrupt", encoding="utf-8")

        cfg.update_config(lambda c: c.__setitem__("n_ctx", 8192))

        got = json.loads(p.read_text(encoding="utf-8"))
        # 8192, not the 4096 DEFAULT: _user_delta drops a value equal to the
        # default, so the default can never appear in the file.
        assert got["n_ctx"] == 8192
        assert got["net_mode"] == "off", "the .bak's real settings were lost"
