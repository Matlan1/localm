# SPDX-License-Identifier: AGPL-3.0-or-later
"""A registry.json whose top level parses as a dict but whose ENTRIES are
malformed (a non-dict value, or a dict with a missing / non-string ``path``)
used to crash every consumer: ``load``/``run``/``sync_models_dir`` all do
``entry["path"]`` / ``entry.get(...)`` assuming the invariant every registry
writer guarantees (``{name: {"path": str, ...}}``). A single hand-edited or
externally-corrupted entry took the whole command down with the generic
"localm hit an unexpected error" bug-report path (AUD-REGSANITIZE).

``load_registry`` / ``update_registry`` now enforce that invariant on read:
malformed entries are dropped (with a stderr warning - do-not-hide-problems),
the good entries survive, and the on-disk file is left intact so the bad entry
stays hand-recoverable (a later atomic write snapshots it to .bak first).
"""

import json

import pytest

import localm.config as cfg


@pytest.fixture()
def reg_home(tmp_path, monkeypatch):
    """Point config's frozen REGISTRY_FILE/CONFIG_FILE at a throwaway home."""
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return home


def _write_registry(home, obj_or_bytes):
    p = home / "registry.json"
    if isinstance(obj_or_bytes, (bytes, bytearray)):
        p.write_bytes(obj_or_bytes)
    else:
        p.write_text(json.dumps(obj_or_bytes), encoding="utf-8")
    return p


GOOD = {"path": "C:/models/good.gguf", "source": "local", "model_type": "llm"}
GOOD_LEGACY = {"path": "/models/legacy.gguf", "source": "local"}  # no model_type


@pytest.mark.parametrize("bad", [
    "not a dict",          # str value
    None,                  # null value
    [1, 2, 3],             # list value
    123,                   # number value
    {"type": "gguf"},      # dict, but no "path"
    {"path": 12345},       # dict, path is an int
    {"path": None},        # dict, path is null
    {"path": ""},          # dict, path is empty string
    {"path": ["x"]},       # dict, path is a list
])
def test_load_registry_drops_malformed_entry_keeps_good(reg_home, bad):
    _write_registry(reg_home, {"good": GOOD, "bad": bad, "legacy": GOOD_LEGACY})
    reg = cfg.load_registry()
    assert reg == {"good": GOOD, "legacy": GOOD_LEGACY}, (
        f"malformed 'bad' entry {bad!r} should be dropped, good ones kept")


def test_load_registry_all_good_untouched(reg_home):
    good = {"a": GOOD, "b": GOOD_LEGACY}
    _write_registry(reg_home, good)
    assert cfg.load_registry() == good


def test_load_registry_leaves_file_on_disk_for_recovery(reg_home):
    """A pure load must NOT rewrite the file - the malformed entry stays
    hand-recoverable; only a later atomic save (which snapshots .bak) drops it."""
    raw = {"good": GOOD, "bad": "corrupt"}
    p = _write_registry(reg_home, raw)
    before = p.read_bytes()
    cfg.load_registry()
    assert p.read_bytes() == before, "load_registry must not rewrite registry.json"


def test_load_registry_warns_on_dropped_entries(reg_home, capsys):
    _write_registry(reg_home, {"good": GOOD, "bad": None, "worse": "x"})
    cfg.load_registry()
    err = capsys.readouterr().err.lower()
    assert "registry" in err and ("malformed" in err or "ignoring" in err), (
        "dropping malformed entries must be surfaced, not silent")


def test_update_registry_sanitizes_and_survives(reg_home):
    """A read-modify-write over a registry with a malformed entry must not crash
    and must not carry the malformed entry forward."""
    _write_registry(reg_home, {"good": GOOD, "bad": "corrupt"})
    out = cfg.update_registry(lambda r: r.__setitem__("new", GOOD))
    assert out.get("good") == GOOD
    assert out.get("new") == GOOD
    assert "bad" not in out
    on_disk = json.loads((reg_home / "registry.json").read_text())
    assert "bad" not in on_disk and "good" in on_disk and "new" in on_disk


def test_sync_models_dir_survives_malformed_entries(reg_home, monkeypatch):
    """sync_models_dir runs on every launch; a malformed registry entry must not
    crash it. It should still register a real loose GGUF sitting in models/."""
    from localm import model_manager as mm

    models = reg_home / "models"
    models.mkdir(parents=True, exist_ok=True)
    # sync_models_dir reads the model_manager package's own MODELS_DIR binding
    # (frozen at import), not config.MODELS_DIR - patch the one it actually reads
    # (module->package patch surface), or it scans the wrong folder under xdist.
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    # A file with the GGUF magic and past the 1 KiB floor so _has_gguf_magic
    # accepts it (sync skips truncated/placeholder .gguf files).
    (models / "real.gguf").write_bytes(b"GGUF" + b"\x00" * 2048)
    _write_registry(reg_home, {"bad": "corrupt", "alsobad": {"path": 7}})

    result = mm.sync_models_dir()  # must not raise
    reg = mm.load_registry()
    assert "real" in reg, "a real loose GGUF should still be discovered"
    assert result.added >= 1


@pytest.mark.parametrize("bad", ["[1, 2, 3]", '"hello"', "42", "null"])
def test_load_config_non_dict_warns_not_silent(reg_home, capsys, bad):
    """A config.json that is valid JSON but not an object (a list/string/number/
    null) must not silently become defaults - the user's settings are being
    ignored, which has to be surfaced (do-not-hide-problems)."""
    (reg_home / "config.json").write_text(bad, encoding="utf-8")
    cfg._warned_bad_config.clear()
    conf = cfg.load_config()
    assert conf["port"] == cfg.DEFAULT_CONFIG["port"]  # falls back to defaults
    err = capsys.readouterr().err.lower()
    assert "config.json" in err and ("not a json object" in err or "ignoring" in err), (
        "a present-but-non-dict config.json must be surfaced, not silently dropped")


def test_load_config_missing_is_silent(reg_home, capsys):
    """A MISSING config.json is the benign default path - no warning."""
    cfg._warned_bad_config.clear()
    assert not (reg_home / "config.json").exists()
    cfg.load_config()
    err = capsys.readouterr().err.lower()
    assert "config.json" not in err, "a missing config is a normal default, not a warning"


def test_load_config_non_dict_leaves_file_for_recovery(reg_home):
    p = reg_home / "config.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    before = p.read_bytes()
    cfg.load_config()
    assert p.read_bytes() == before, "load_config must not rewrite a non-dict config.json"


def test_list_cli_survives_malformed_registry(cli_runner):
    """End-to-end: `localm list` must not hit the crash / bug-report path on a
    registry with malformed entries."""
    from localm.cli import main
    from localm.config import REGISTRY_FILE

    REGISTRY_FILE.write_text(json.dumps({
        "bad": "not a dict",
        "worse": None,
        "int_path": {"path": 999},
    }), encoding="utf-8")

    result = cli_runner.invoke(main, ["list"])
    assert result.exit_code == 0, (
        f"list crashed on malformed registry: {result.output}\n{result.exception}")
    assert "unexpected error" not in result.output.lower()
