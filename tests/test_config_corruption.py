# SPDX-License-Identifier: AGPL-3.0-or-later
"""A config.json that parses as valid JSON but is NOT an object (a list, a bare
string, a number, null) used to be silently ignored -> defaults, discarding the
user's saved settings with no warning. (A genuinely unparseable file DID warn via
_read_json; only the valid-JSON-wrong-shape case was silent.)

`_merge_stored_config` now surfaces that discard once per process, while a MISSING
file stays the benign silent default (AUD-CFGNONDICT)."""

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
