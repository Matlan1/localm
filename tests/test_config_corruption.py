# SPDX-License-Identifier: AGPL-3.0-or-later
"""A config.json that parses as valid JSON but is NOT an object (a list, a bare
string, a number, null) resolves to defaults, discarding the user's saved
settings. A genuinely unparseable file warns via _read_json; the
valid-JSON-wrong-shape case would otherwise be silent.

`_merge_stored_config` surfaces that discard once per process, while a MISSING
file stays the benign silent default."""

import os

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
    """``config.os`` as seen on another platform: ``name`` replaced, the rest
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
    monkeypatch.setattr(cfg, "time", types.SimpleNamespace(
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
    monkeypatch.setattr(cfg, "open", fake_open, raising=False)
    return calls


def test_atomic_write_private_uses_a_bare_replace_by_default(cfg_home, monkeypatch):
    retried = []
    monkeypatch.setattr(cfg, "_replace_atomic", lambda src, dst: retried.append(dst))
    target = cfg_home / "secret.json"
    cfg.atomic_write_private(target, "x")
    assert retried == []
    assert target.read_text(encoding="utf-8") == "x"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file mode bits")
def test_atomic_write_private_creates_its_temp_file_owner_only(cfg_home, monkeypatch):
    seen = []
    real = cfg.restrict_file_perms

    def spy(path, **kw):
        seen.append(os.stat(path).st_mode & 0o777)
        return real(path, **kw)
    monkeypatch.setattr(cfg, "restrict_file_perms", spy)
    old = os.umask(0)
    try:
        cfg.atomic_write_private(cfg_home / "secret.json", "x")
    finally:
        os.umask(old)
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
    monkeypatch.setattr(cfg, "os", proxy)
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


def test_read_json_checked_reports_a_corrupt_file(cfg_home, capsys):
    target = cfg_home / "x.json"
    target.write_text("{bad", encoding="utf-8")
    assert cfg._read_json_checked(target, {}) == ({}, False)
    err = capsys.readouterr().err
    assert err.startswith("[localm] x.json is unreadable (")
    assert err.endswith("); falling back.\n")


def test_read_json_checked_on_windows_retries_a_transient_denial_then_reports(
        cfg_home, monkeypatch, capsys):
    target = cfg_home / "x.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cfg, "os", _OsAs("nt"))
    sleeps = _record_sleeps(monkeypatch)
    calls = _deny_open(monkeypatch, target, 32)
    assert cfg._read_json_checked(target, {}) == ({}, False)
    assert len(calls) == cfg._REPLACE_RETRIES
    assert len(sleeps) == cfg._REPLACE_RETRIES - 1
    assert capsys.readouterr().err == (
        f"[localm] x.json is unreadable ({calls[-1]}); falling back.\n")


def test_read_json_checked_on_windows_does_not_retry_a_stable_denial(
        cfg_home, monkeypatch, capsys):
    target = cfg_home / "x.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cfg, "os", _OsAs("nt"))
    sleeps = _record_sleeps(monkeypatch)
    calls = _deny_open(monkeypatch, target, 1314)
    assert cfg._read_json_checked(target, {}) == ({}, False)
    assert len(calls) == 1 and sleeps == []
    assert capsys.readouterr().err == (
        f"[localm] x.json is unreadable ({calls[-1]}); falling back.\n")


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
    ("ON", "on"), ("sideways", None),
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
