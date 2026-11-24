# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``localm info``, ``localm config`` and ``localm list`` print: aligned info
rows, no credential in clear, and the real file format of every model."""

from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

import localm.cli.models as modelscli
import localm.model_manager as mm
from localm import config
from localm.cli import main
from localm.model_manager import registry as registry_mod
from localm.settings_schema import CORE_FIELDS, Widget


@pytest.fixture
def info(monkeypatch, tmp_path):
    monkeypatch.setenv("COLUMNS", "400")

    def run(**values):
        monkeypatch.setattr(modelscli, "HOME_DIR", tmp_path)
        monkeypatch.setattr(modelscli, "load_config",
                            lambda: {**config.DEFAULT_CONFIG, **values})
        res = CliRunner().invoke(main, ["info"])
        assert res.exit_code == 0, res.output
        return res.output

    return run


def test_every_header_value_starts_in_the_same_column(info):
    lines = info().splitlines()
    starts = {}
    for label in ("models dir", "registry", "config", "binaries", "install", "temp dir"):
        line = next(ln for ln in lines if ln.strip().startswith(label))
        rest = line[line.index(label) + len(label):]
        starts[label] = len(line) - len(rest.lstrip())
    assert len(set(starts.values())) == 1, starts


def test_info_shows_no_credential(info):
    out = info(update_token="UPD-SECRET", bugreport_upload_token="BUG-SECRET",
               hf_token="HF-SECRET", my_api_key="KEY-SECRET", tls_key="D:/certs/key.pem",
               net_search_url="https://me:URL-SECRET@host/s?api_key=QUERY-SECRET&q=x",
               plugins={"x": {"password": "NEST-SECRET", "depth": 3,
                              "servers": [{"auth_token": "LIST-SECRET", "host": "h1"}]}})
    assert "SECRET" not in out
    for shown in ("update_token           (set)", "'password': '(set)'", "'auth_token': '(set)'",
                  "'depth': 3", "'host': 'h1'", "https://<redacted>@host/s?api_key=<redacted>&q=x",
                  "D:/certs/key.pem"):
        assert shown in out


def test_info_says_when_a_credential_is_not_set(info):
    assert "bugreport_upload_token (not set)" in info(bugreport_upload_token="")


def test_a_setting_named_like_a_credential_but_holding_none_is_shown(info):
    out = info(require_auth=True)
    assert "require_auth           True" in out
    assert f"max_tokens             {config.DEFAULT_CONFIG['max_tokens']}" in out


def test_only_the_token_settings_of_the_defaults_count_as_credentials():
    assert {k for k, v in config.DEFAULT_CONFIG.items()
            if modelscli._is_credential_setting(k) and not isinstance(v, bool)} == {
                "bugreport_upload_token", "update_token"}


def test_every_schema_secret_is_a_credential_setting_whatever_its_name(monkeypatch):
    monkeypatch.setattr(modelscli, "_CREDENTIAL_NAME_RE", re.compile(r"(?!)"))
    secrets = [f.key for f in CORE_FIELDS if f.secret or f.widget == Widget.SECRET]
    assert secrets and all(modelscli._is_credential_setting(k) for k in secrets)


def test_setting_a_token_does_not_print_it(monkeypatch):
    cfg: dict = {}
    monkeypatch.setattr(modelscli, "update_config", lambda fn: fn(cfg))
    res = CliRunner().invoke(main, ["config", "update_token", "UPD-SECRET"])
    assert res.exit_code == 0, res.output
    assert cfg["update_token"] == "UPD-SECRET"
    assert "SECRET" not in res.output and "update_token = (set)" in res.output


@pytest.mark.parametrize("name, head, shown", [
    ("RealESRGAN_x4plus.pth", b"PK\x03\x04" + b"\0" * 40, "pth"),
    ("v1-5.ckpt", b"PK\x03\x04" + b"\0" * 40, "ckpt"),
    ("a.safetensors", b"\x08\0\0\0\0\0\0\0{}", "safetensors"),
    ("a.sft", b"\x08\0\0\0\0\0\0\0{}", "safetensors"),
    ("m.gguf", b"GGUF", "gguf"),
    ("M.GGUF", b"GGUF", "gguf"),
    ("sha256-" + "a" * 64, b"GGUF" + b"\0" * 40, "gguf"),
    ("noextension", b"xxxx", "file"),
])
def test_list_shows_the_format_of_the_file(monkeypatch, tmp_path, name, head, shown):
    path = tmp_path / name
    path.write_bytes(head)
    assert registry_mod._file_format(path) == shown
    monkeypatch.setattr(mm, "load_registry", lambda: {
        "entry": {"path": str(path), "source": "local", "model_type": "upscaler"}})
    rows: list = []
    monkeypatch.setattr(registry_mod.Table, "add_row", lambda self, *cells, **kw: rows.append(cells))
    registry_mod.list_models()
    assert rows[0][1] == shown
