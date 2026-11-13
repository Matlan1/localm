# SPDX-License-Identifier: AGPL-3.0-or-later
"""A config.json that parses as valid JSON but is NOT an object (a list, a bare
string, a number, null) resolves to defaults, discarding the user's saved
settings. A genuinely unparseable file warns via _read_json; the
valid-JSON-wrong-shape case would otherwise be silent.

`_merge_stored_config` surfaces that discard once per process, while a MISSING
file stays the benign silent default."""

import json
import os

import pytest

import localm.config as cfg
import localm.storekit as storekit


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
        assert "registry.json" in str(raised)
        assert "could not be read" in str(raised)
        assert str(cfg_home) not in str(raised)

    def test_absent_config_still_writes(self, cfg_home):
        """The control: first run must be unaffected, or the refusal would be
        unfalsifiable (a fix that refused everything would pass the tests above)."""
        p = cfg_home / "config.json"
        assert not p.is_file()
        cfg.update_config(lambda c: c.__setitem__("port", 9191))
        import json
        assert json.loads(p.read_text(encoding="utf-8"))["port"] == 9191

    def test_a_primary_served_from_bak_is_refused_too(self, cfg_home):
        """An unreadable PRIMARY whose .bak reads fine is the previous
        generation, not the current settings: saving would persist the older
        copy over it, so it refuses and leaves both files as they were."""
        import json
        p = cfg_home / "config.json"
        p.write_text(json.dumps({"net_mode": "off"}), encoding="utf-8")
        cfg.update_config(lambda c: c.__setitem__("port", 9393))   # rotates .bak
        p.write_text("{ corrupt", encoding="utf-8")
        bak = p.with_name("config.json.bak")
        bak_before = bak.read_bytes()

        with pytest.raises(cfg.ConfigUnreadable):
            cfg.update_config(lambda c: c.__setitem__("n_ctx", 8192))

        assert p.read_text(encoding="utf-8") == "{ corrupt"
        assert bak.read_bytes() == bak_before


# --------------------------------------------------------------------------- #
#  the config-writing routes answer an unreadable config.json with a 409      #
# --------------------------------------------------------------------------- #

_OWNER_KEY = "owner-admin-key-config-corruption-409"


@pytest.fixture()
def owner_client(cfg_home, monkeypatch):
    """The real app with an owner key; unhandled server errors come back as a
    500 response instead of being raised into the test."""
    from fastapi.testclient import TestClient

    from localm.inference.http_server import create_app
    monkeypatch.setenv("LOCALM_API_KEY", _OWNER_KEY)
    app = create_app(None)
    with TestClient(app, raise_server_exceptions=False,
                    headers={"Authorization": f"Bearer {_OWNER_KEY}"}) as c:
        yield c


@pytest.mark.parametrize("method,url,body", [
    ("PATCH", "/v1/config", {"import_max_depth": 5}),
    ("POST", "/v1/media/config/image", {"delete_outputs": True}),
    ("POST", "/v1/tts/config", {"voice": "am_onyx"}),
])
def test_config_route_answers_an_unreadable_config_with_409(
        owner_client, cfg_home, method, url, body):
    p = cfg_home / "config.json"
    p.write_bytes(b"{ this is not json")
    (cfg_home / "config.json.bak").unlink(missing_ok=True)
    corrupt = p.read_bytes()

    r = owner_client.request(method, url, json=body)

    assert p.read_bytes() == corrupt
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "config.json" in detail
    assert str(cfg_home) not in detail


# --------------------------------------------------------------------------- #
#  JSON store reads and writes                                                #
# --------------------------------------------------------------------------- #

class _OsAs:
    """``storekit.os`` as seen on another platform: ``name`` replaced, the rest
    from the real module."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        import os
        return getattr(os, attr)


def _denial(winerror):
    e = PermissionError(13, "injected denial")
    e.winerror = winerror
    return e


def _record_sleeps(monkeypatch):
    import time
    import types
    sleeps = []
    monkeypatch.setattr(storekit, "time", types.SimpleNamespace(
        sleep=sleeps.append, time=time.time, monotonic=time.monotonic,
        strftime=time.strftime))
    return sleeps


def _deny_open(monkeypatch, path, winerror):
    import builtins
    calls = []

    def fake_open(file, *a, **k):
        if str(file) == str(path):
            calls.append(_denial(winerror))
            raise calls[-1]
        return builtins.open(file, *a, **k)
    monkeypatch.setattr(storekit, "open", fake_open, raising=False)
    return calls


def test_atomic_write_private_uses_a_bare_replace_by_default(cfg_home, monkeypatch):
    retried = []
    monkeypatch.setattr(storekit, "replace_atomic", lambda src, dst: retried.append(dst))
    target = cfg_home / "secret.json"
    cfg.atomic_write_private(target, "x")
    assert retried == []
    assert target.read_text(encoding="utf-8") == "x"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode bits")
def test_atomic_write_private_creates_its_temp_file_owner_only(cfg_home, monkeypatch):
    seen = []
    real = storekit.restrict_file_perms

    def spy(path, **kw):
        seen.append(os.stat(path).st_mode & 0o777)
        return real(path, **kw)
    monkeypatch.setattr(storekit, "restrict_file_perms", spy)
    cfg.atomic_write_private(cfg_home / "secret.json", "x")
    assert seen[0] == 0o600


def test_atomic_write_private_completes_a_short_write(cfg_home, monkeypatch):
    text = "abcdefghij" * 3
    calls = []

    def write(fd, data):
        calls.append(len(data))
        if len(calls) > len(text):
            raise AssertionError("the write loop did not advance")
        return os.write(fd, bytes(data[:3]))
    proxy = _OsAs(os.name)
    proxy.write = write
    monkeypatch.setattr(storekit, "os", proxy)
    target = cfg_home / "secret.json"
    cfg.atomic_write_private(target, text)
    assert target.read_bytes() == text.encode("utf-8")
    assert calls == list(range(len(text), 0, -3))


def test_atomic_write_json_uses_a_unique_sibling_temp_and_indents(cfg_home, monkeypatch):
    import json
    import re
    target = cfg_home / "x.json"
    moves = []
    real = cfg._replace_atomic

    def spy(src, dst):
        moves.append((src, dst))
        real(src, dst)
    monkeypatch.setattr(cfg, "_replace_atomic", spy)
    cfg._atomic_write_json(target, {"a": [1, 2]})
    [(src, dst)] = moves
    assert dst == target
    assert src.parent == target.parent
    assert re.fullmatch(r"x\.json\.\w+\.tmp", src.name), src.name
    assert target.read_text(encoding="utf-8") == json.dumps({"a": [1, 2]}, indent=2)


def test_atomic_write_json_keeps_an_unreadable_file_under_a_dated_name(cfg_home, capsys):
    import re
    target = cfg_home / "x.json"
    target.write_text("{bad", encoding="utf-8")
    cfg._atomic_write_json(target, {"a": 1})
    [kept] = [p for p in cfg_home.iterdir() if ".unreadable-" in p.name]
    assert re.fullmatch(r"x\.json\.unreadable-\d{8}-\d{6}", kept.name), kept.name
    assert kept.read_text(encoding="utf-8") == "{bad"
    out = capsys.readouterr()
    assert out.err == (f"[localm] x.json could not be read; kept it as {kept.name} "
                       "and left x.json.bak unchanged.\n")
    assert out.out == ""


def test_holds_unparseable_json(cfg_home):
    bad = cfg_home / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    good = cfg_home / "good.json"
    good.write_text("{}", encoding="utf-8")
    assert cfg._holds_unparseable_json(bad) is True
    assert cfg._holds_unparseable_json(good) is False
    assert cfg._holds_unparseable_json(cfg_home / "missing.json") is False


def test_read_json_checked_falls_back_to_the_bak_and_calls_a_default(cfg_home):
    target = cfg_home / "x.json"
    assert cfg._read_json_checked(target, dict) == ({}, True)
    (cfg_home / "x.json.bak").write_text('{"a": 1}', encoding="utf-8")
    assert cfg._read_json_checked(target, dict) == ({"a": 1}, True)


def test_read_json_checked_reports_a_corrupt_file(cfg_home, capsys, caplog):
    import logging
    target = cfg_home / "x.json"
    target.write_text("{bad", encoding="utf-8")
    storekit._warned_unreadable.clear()
    with caplog.at_level(logging.WARNING, logger="localm"):
        assert cfg._read_json_checked(target, {}) == ({}, False)
    assert capsys.readouterr().err == ""
    [msg] = [r.getMessage() for r in caplog.records if "x.json" in r.getMessage()]
    assert msg.startswith("x.json is unreadable (") and msg.endswith(")")


def test_read_json_checked_on_windows_retries_a_transient_denial_then_reports(
        cfg_home, monkeypatch, capsys, caplog):
    import logging
    storekit._warned_unreadable.clear()
    caplog.set_level(logging.WARNING, logger="localm")
    target = cfg_home / "x.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(storekit, "os", _OsAs("nt"))
    sleeps = _record_sleeps(monkeypatch)
    calls = _deny_open(monkeypatch, target, 32)
    assert cfg._read_json_checked(target, {}) == ({}, False)
    assert len(calls) == storekit._REPLACE_RETRIES
    assert len(sleeps) == storekit._REPLACE_RETRIES - 1
    assert capsys.readouterr().err == ""
    assert [r.getMessage() for r in caplog.records if "x.json" in r.getMessage()] == [
        f"x.json is unreadable ({calls[-1]})"]


def test_read_json_checked_on_windows_does_not_retry_a_stable_denial(
        cfg_home, monkeypatch, capsys, caplog):
    import logging
    storekit._warned_unreadable.clear()
    caplog.set_level(logging.WARNING, logger="localm")
    target = cfg_home / "x.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(storekit, "os", _OsAs("nt"))
    sleeps = _record_sleeps(monkeypatch)
    calls = _deny_open(monkeypatch, target, 1314)
    assert cfg._read_json_checked(target, {}) == ({}, False)
    assert len(calls) == 1 and sleeps == []
    assert capsys.readouterr().err == ""
    assert [r.getMessage() for r in caplog.records if "x.json" in r.getMessage()] == [
        f"x.json is unreadable ({calls[-1]})"]


def test_non_dict_config_warning_is_once_per_config_file(cfg_home, monkeypatch, capsys):
    monkeypatch.setattr(cfg, "_warned_bad_config", set())
    merged: dict = {}
    cfg._merge_stored_config(merged, [1])
    cfg._merge_stored_config(merged, [1])
    assert capsys.readouterr().err == (
        "[localm] config.json is not a JSON object (got list); ignoring it and "
        "using defaults. The file is left untouched so you can recover it by "
        "hand.\n")
    monkeypatch.setattr(cfg, "CONFIG_FILE", cfg_home / "other" / "config.json")
    cfg._merge_stored_config(merged, "text")
    assert "(got str)" in capsys.readouterr().err
    assert merged == {}


def test_update_config_of_an_absent_file_is_silent_and_leaves_defaults_alone(
        cfg_home, monkeypatch, capsys):
    monkeypatch.setattr(cfg, "_warned_bad_config", set())
    cfg.update_config(lambda c: c["plugins"].update({"probe": {"x": 1}}))
    assert "probe" not in cfg.DEFAULT_CONFIG["plugins"]
    assert cfg.load_config()["plugins"]["probe"] == {"x": 1}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("raw, want", [
    (True, "on"), (False, "off"), (1, "on"), (0, "off"), (" ASK ", "ask"),
    ("Off", "off"), ("yes", "on"),
])
def test_remote_image_mode_is_normalised_in_place(cfg_home, raw, want):
    stored = {"gui_proxy_remote_images": raw}
    cfg._normalize_remote_image_mode(stored)
    assert stored == {"gui_proxy_remote_images": want}


@pytest.mark.parametrize("raw, want", [
    (1, "on"), (0, "off"), (2, None), (5, None), (1.0, None), (" ASK ", "ask"),
    ("ON", "on"), ("sideways", None), ("1", "on"), ("TRUE", "on"), ("yes", "on"),
    ("0", "off"), ("False", "off"), ("NO", "off"),
])
def test_coerce_remote_image_mode(raw, want):
    assert cfg.coerce_remote_image_mode(raw) == want


@pytest.mark.parametrize("env, stored, want", [
    (None, False, False), (None, True, True), ("TRUE", False, True),
    ("1", False, True), ("true", False, True), ("on", False, True),
    ("no", False, False),
])
def test_keep_diagnostics_enabled(cfg_home, monkeypatch, env, stored, want):
    if env is None:
        monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    else:
        monkeypatch.setenv("LOCALM_KEEP_DIAGNOSTICS", env)
    cfg.update_config(lambda c: c.update(keep_diagnostics=stored))
    assert cfg.keep_diagnostics_enabled() is want


def test_detect_home_treats_unset_and_blank_localm_home_alike(monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", "   ")
    blank = cfg._detect_home()
    monkeypatch.delenv("LOCALM_HOME")
    assert cfg._detect_home() == blank


def _fake_install(tmp_path, monkeypatch):
    root = tmp_path / "install"
    (root / "localm").mkdir(parents=True)
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(cfg, "__file__", str(root / "localm" / "config.py"))
    monkeypatch.delenv("LOCALM_HOME", raising=False)
    monkeypatch.delenv("LOCALM_SETUP", raising=False)
    monkeypatch.setattr(cfg, "_warned_unconfigured_home", False)
    return root.resolve()


def test_detect_home_reads_the_install_marker_file(tmp_path, monkeypatch):
    root = _fake_install(tmp_path, monkeypatch)
    target = tmp_path / "data"
    (root / "localm-home.cfg").write_text(f"{target}\n", encoding="utf-8")
    assert cfg._detect_home() == target


def test_detect_home_ignores_an_empty_install_marker_file(tmp_path, monkeypatch):
    root = _fake_install(tmp_path, monkeypatch)
    (root / "localm-home.cfg").write_text("\n", encoding="utf-8")
    assert cfg._detect_home() == root / "home"


def test_detect_home_uses_a_portable_home_folder(tmp_path, monkeypatch, capsys):
    root = _fake_install(tmp_path, monkeypatch)
    (root / "home").mkdir()
    assert cfg._detect_home() == root / "home"
    assert capsys.readouterr().err == ""


def test_detect_home_falls_back_to_a_contained_home_and_says_so(tmp_path, monkeypatch, capsys):
    root = _fake_install(tmp_path, monkeypatch)
    assert cfg._detect_home() == root / "home"
    assert str(root / "home") in capsys.readouterr().err


def test_unconfigured_home_warning_is_printed_once(tmp_path, monkeypatch, capsys):
    _fake_install(tmp_path, monkeypatch)
    cfg._warn_unconfigured_home(tmp_path / "h")
    cfg._warn_unconfigured_home(tmp_path / "h")
    out = capsys.readouterr()
    assert out.err == (
        "[localm] WARNING: no data directory is configured (no LOCALM_HOME, no "
        "localm-home.cfg, no ./home). Setup may not have run, or the config was "
        f"lost. Using a contained default inside this install: {tmp_path / 'h'}. "
        "Run setup or set LOCALM_HOME to choose where data lives.\n")
    assert out.out == ""


def test_unconfigured_home_warning_is_silent_during_setup(tmp_path, monkeypatch, capsys):
    _fake_install(tmp_path, monkeypatch)
    monkeypatch.setenv("LOCALM_SETUP", "1")
    cfg._warn_unconfigured_home(tmp_path / "h")
    assert capsys.readouterr().err == ""


def test_ensure_dirs_explains_a_file_data_dir_without_localm_home(tmp_path, monkeypatch):
    import click
    home_file = tmp_path / "home-file"
    home_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cfg, "HOME_DIR", home_file)
    monkeypatch.delenv("LOCALM_HOME", raising=False)
    with pytest.raises(click.ClickException) as err:
        cfg.ensure_dirs()
    assert err.value.message.startswith("localm's data path is a file")


def test_ensure_dirs_names_localm_home_only_for_the_home_dir(tmp_path, monkeypatch):
    import click
    home_file = tmp_path / "home-file"
    home_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cfg, "HOME_DIR", home_file)
    monkeypatch.setenv("LOCALM_HOME", str(home_file))
    with pytest.raises(click.ClickException) as home_err:
        cfg.ensure_dirs()
    assert home_err.value.message.startswith("LOCALM_HOME points at a file")
    models_file = tmp_path / "models-file"
    models_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", models_file)
    monkeypatch.setenv("LOCALM_HOME", str(models_file))
    with pytest.raises(click.ClickException) as models_err:
        cfg.ensure_dirs()
    assert models_err.value.message.startswith("localm's data path is a file")


def test_peek_registry_returns_the_registered_models(cfg_home):
    registry = {"tiny": {"path": "models/tiny.gguf", "format": "gguf"}}
    (cfg_home / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    assert cfg.peek_registry() == registry


@pytest.mark.parametrize("bad", ["[1, 2, 3]", "null", "{not json"])
def test_peek_registry_of_a_non_object_registry_is_empty(cfg_home, bad):
    (cfg_home / "registry.json").write_text(bad, encoding="utf-8")

    assert cfg.peek_registry() == {}


def test_peek_registry_creates_no_directories(tmp_path, monkeypatch):
    home = tmp_path / "absent-home"
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")

    assert cfg.peek_registry() == {}
    assert not home.exists()


# --------------------------------------------------------------------------- #
#  registry IDs, base names and the model index                               #
# --------------------------------------------------------------------------- #

def _write_registry(home, reg):
    (home / "registry.json").write_text(json.dumps(reg), encoding="utf-8")


def _stored_registry(home):
    return json.loads((home / "registry.json").read_text(encoding="utf-8"))


def _renamed_base_registry():
    """A base entry renamed from ``old-base`` to ``base`` and a fine-tune that
    still names it ``old-base``, every entry with an ID."""
    return {
        "base": {"id": "id-base", "path": "b.gguf", "former_names": ["old-base"]},
        "tune": {"id": "id-tune", "path": "t.gguf", "base": "old-base"},
    }


def test_load_registry_follows_a_renamed_base(cfg_home):
    _write_registry(cfg_home, _renamed_base_registry())

    reg = cfg.load_registry()

    assert reg["tune"]["base"] == "base"
    assert reg["tune"]["model_refs"] == {"base": "id-base"}


def test_load_registry_gives_entries_ids_and_writes_them_back(cfg_home):
    _write_registry(cfg_home, {"m": {"path": "m.gguf"}})

    reg = cfg.load_registry()

    assert isinstance(reg["m"].get("id"), str) and reg["m"]["id"]
    assert _stored_registry(cfg_home)["m"]["id"] == reg["m"]["id"]


def test_peek_registry_follows_a_renamed_base(cfg_home):
    _write_registry(cfg_home, _renamed_base_registry())

    reg = cfg.peek_registry()

    assert reg["tune"]["base"] == "base"
    assert reg["tune"]["model_refs"] == {"base": "id-base"}


def test_save_registry_gives_ids_and_follows_a_renamed_base(cfg_home):
    reg = _renamed_base_registry()
    reg["new"] = {"path": "n.gguf"}

    cfg.save_registry(reg)

    stored = _stored_registry(cfg_home)
    assert isinstance(stored["new"].get("id"), str) and stored["new"]["id"]
    assert stored["tune"]["base"] == "base"
    assert stored["tune"]["model_refs"] == {"base": "id-base"}


def test_update_registry_keeps_existing_entries(cfg_home):
    _write_registry(cfg_home, {"a": {"id": "id-a", "path": "a.gguf"}})

    cfg.update_registry(lambda r: r.__setitem__("b", {"path": "b.gguf"}))

    stored = _stored_registry(cfg_home)
    assert stored["a"] == {"id": "id-a", "path": "a.gguf"}
    assert stored["b"]["path"] == "b.gguf"


def test_update_registry_mutator_sees_ids_and_current_base_names(cfg_home):
    reg = _renamed_base_registry()
    reg["fresh"] = {"path": "f.gguf"}
    _write_registry(cfg_home, reg)
    seen = {}

    def mutator(r):
        seen["fresh_id"] = r["fresh"].get("id")
        seen["tune_base"] = r["tune"].get("base")

    out = cfg.update_registry(mutator)

    assert isinstance(seen["fresh_id"], str) and seen["fresh_id"]
    assert out["fresh"]["id"] == seen["fresh_id"]
    assert _stored_registry(cfg_home)["fresh"]["id"] == seen["fresh_id"]
    assert seen["tune_base"] == "base"


def test_update_registry_entry_replaced_under_its_name_keeps_id_and_former_names(cfg_home):
    _write_registry(cfg_home, {
        "m": {"id": "id-m", "path": "old.gguf", "former_names": ["older"]}})

    cfg.update_registry(lambda r: r.__setitem__("m", {"path": "new.gguf"}))

    stored = _stored_registry(cfg_home)
    assert stored["m"] == {"id": "id-m", "path": "new.gguf", "former_names": ["older"]}


def test_update_registry_points_bases_at_a_model_renamed_by_the_mutator(cfg_home):
    from localm import model_ids
    _write_registry(cfg_home, {
        "b": {"id": "id-b", "path": "b.gguf"},
        "tune": {"id": "id-tune", "path": "t.gguf", "base": "b"},
    })

    cfg.update_registry(lambda r: model_ids.rename_entry(r, "b", "b2"))

    stored = _stored_registry(cfg_home)
    assert stored["tune"]["base"] == "b2"
    assert stored["tune"]["model_refs"] == {"b2": "id-b"}


@pytest.fixture()
def fresh_model_index(monkeypatch):
    monkeypatch.setattr(cfg, "_model_index_cache", None)


def test_model_index_reads_the_registry(cfg_home, fresh_model_index):
    _write_registry(cfg_home, _renamed_base_registry())

    index = cfg.model_index()

    assert index.by_name == {"base": "id-base", "tune": "id-tune"}
    assert index.former == {"old-base": "base"}


def test_model_index_of_an_unchanged_registry_is_served_from_the_cache(
        cfg_home, fresh_model_index):
    _write_registry(cfg_home, _renamed_base_registry())

    first = cfg.model_index()

    assert cfg.model_index() is first


def test_model_index_follows_a_changed_registry(cfg_home, fresh_model_index):
    _write_registry(cfg_home, {"a": {"id": "id-a"}})
    assert cfg.model_index().by_name == {"a": "id-a"}

    _write_registry(cfg_home, {"a": {"id": "id-a"}, "bb": {"id": "id-bb"}})

    assert cfg.model_index().by_name == {"a": "id-a", "bb": "id-bb"}


def test_model_index_tells_two_registry_files_apart(
        cfg_home, fresh_model_index, monkeypatch):
    first = cfg_home / "one" / "registry.json"
    second = cfg_home / "two" / "registry.json"
    stamp = 1_700_000_000_000_000_000
    for path, name in ((first, "aa"), (second, "bb")):
        path.parent.mkdir()
        path.write_text(json.dumps({name: {"id": "x"}}), encoding="utf-8")
        os.utime(path, ns=(stamp, stamp))
    assert first.stat().st_size == second.stat().st_size
    assert first.stat().st_mtime_ns == second.stat().st_mtime_ns

    monkeypatch.setattr(cfg, "REGISTRY_FILE", first)
    assert cfg.model_index().by_name == {"aa": "x"}
    monkeypatch.setattr(cfg, "REGISTRY_FILE", second)
    assert cfg.model_index().by_name == {"bb": "x"}


# --------------------------------------------------------------------------- #
#  lexical UNC / device path test and config migration helpers                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("os_name, text, want", [
    ("nt", "\\\\server\\share", True),
    ("posix", "\\\\server\\share", True),
    ("nt", "//server/share", True),
    ("posix", "//server/share", False),
    ("nt", "\\??\\C:\\x", True),
    ("posix", "\\??\\C:\\x", False),
    ("nt", "C:\\data", False),
    ("posix", "/data", False),
])
def test_is_unc_or_device(monkeypatch, os_name, text, want):
    monkeypatch.setattr(cfg, "os", _OsAs(os_name))

    assert cfg._is_unc_or_device(text) is want


@pytest.mark.parametrize("raw, want", [
    (True, "on"), (False, "off"), ("YES", "on"), ("sideways", "sideways"), (2, 2),
])
def test_remote_image_migration_maps_legacy_values_and_keeps_unknown_ones(raw, want):
    assert cfg._map_remote_image_value(raw) == want
    stored = {"gui_proxy_remote_images": raw}
    cfg._apply_config_migrations(stored)
    assert stored["gui_proxy_remote_images"] == want


def test_config_path_parent_walks_to_the_parent_of_the_last_key():
    root = {"a": {"b": {"c": 1}}, "x": 5}

    assert cfg._config_path_parent(root, ("x",)) is root
    assert cfg._config_path_parent(root, ("a", "b")) is root["a"]
    assert cfg._config_path_parent(root, ("a", "b", "c")) is root["a"]["b"]
    assert cfg._config_path_parent(root, ("x", "y")) is None


@pytest.mark.parametrize("stored, want", [
    ("false", False), ("0", False), ("no", False), ("off", False), ("", False),
    ("garbage", False), (0, False), (1, False), (None, False), ([], False),
    ({"on": True}, False), (True, True), ("true", True), (" Yes ", True), ("1", True),
])
def test_keep_diagnostics_reads_only_an_explicit_on(cfg_home, monkeypatch, stored, want):
    """A hand-edited value turns diagnostics on only when it says on."""
    monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    cfg.update_config(lambda c: c.update(keep_diagnostics=stored))
    assert cfg.keep_diagnostics_enabled() is want


def test_an_unrecognised_keep_diagnostics_value_is_reported_once(cfg_home, monkeypatch, capsys):
    monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(cfg, "_warned_bad_keep_diagnostics", set())
    cfg.update_config(lambda c: c.update(keep_diagnostics="false-ish"))
    assert cfg.keep_diagnostics_enabled() is False
    assert cfg.keep_diagnostics_enabled() is False
    err = capsys.readouterr().err
    assert err.count("keep_diagnostics='false-ish'") == 1


def test_privacy_mode_with_a_string_false_keeps_diagnostics_off(cfg_home, monkeypatch):
    from localm.audit import diagnostics_allowed
    monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("LOCALM_MODE", raising=False)
    cfg.update_config(lambda c: c.update(keep_diagnostics="false", mode="privacy"))
    assert diagnostics_allowed() is False


@pytest.mark.parametrize("stored, want", [("false", False), ("garbage", False), (1, False),
                                          ("yes", True), (True, True)])
def test_load_config_and_the_settings_schema_read_keep_diagnostics_as_a_bool(
        cfg_home, monkeypatch, stored, want):
    """The settings page renders the schema value, so it must show what is in force."""
    from localm.settings_schema import schema_json
    monkeypatch.delenv("LOCALM_KEEP_DIAGNOSTICS", raising=False)
    raw = json.loads(cfg.CONFIG_FILE.read_text(encoding="utf-8")) if cfg.CONFIG_FILE.exists() else {}
    raw["keep_diagnostics"] = stored
    cfg.CONFIG_FILE.write_text(json.dumps(raw), encoding="utf-8")
    assert cfg.load_config()["keep_diagnostics"] is want
    fields = [f for f in schema_json(cfg.load_config()) if f["key"] == "keep_diagnostics"]
    assert fields and fields[0]["default"] is want
