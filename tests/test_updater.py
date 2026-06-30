# SPDX-License-Identifier: AGPL-3.0-or-later
"""The self-updater: configured endpoint resolution, the version check, update
classification (reboot/deps/runtime/setup), and the build download. Apply is NOT
exercised here (it has its own detached-helper tests)."""

import json

import pytest

from localm import _version, updater


def _opener(payload, status=200):
    """A _proxy.request opener: (method, url, data, headers, timeout) -> (status, bytes)."""
    def op(method, url, data, headers, timeout):
        op.url = url
        op.token = headers.get("X-Localm-Token")
        return status, json.dumps(payload).encode("utf-8")
    return op


def _write_pyproject(d, deps, *, extras=None, requires_python="==3.12.*"):
    lines = ['[project]', 'name = "localm"', 'version = "0.1.0"',
             f'requires-python = "{requires_python}"',
             'dependencies = [' + ", ".join(f'"{x}"' for x in deps) + ']']
    content = "\n".join(lines) + "\n"
    if extras:
        content += "[project.optional-dependencies]\n"
        for name, items in extras.items():
            content += f'{name} = [' + ", ".join(f'"{x}"' for x in items) + ']\n'
    (d / "pyproject.toml").write_text(content, encoding="utf-8")


# ------------------------------- endpoint -------------------------------

def test_endpoint_reuses_bugreport_config(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w.example", "bugreport_upload_token": "t"})
    assert updater.endpoint() == ("https://w.example", "t")
    assert updater.available() is True


def test_endpoint_update_override_wins(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://bug", "bugreport_upload_token": "bt",
        "update_url": "https://upd", "update_token": "ut"})
    assert updater.endpoint() == ("https://upd", "ut")


def test_unconfigured(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert updater.available() is False


# -------------------------------- check ---------------------------------

def test_check_reports_newer(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _opener({"ok": True, "version": "v0.2.0", "notes": "stuff",
                  "asset": {"id": 3, "name": "localm-0.2.0.zip", "size": 9}})
    res = updater.check(opener=op)
    assert res["current"] == "0.1.0"
    assert res["latest"] == "v0.2.0"
    assert res["newer"] is True
    assert res["asset"]["id"] == 3
    assert op.url.endswith("/update")


def test_check_not_newer_when_same(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    res = updater.check(opener=_opener({"version": "v0.2.0"}))
    assert res["newer"] is False


def test_check_no_releases(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    res = updater.check(opener=_opener({"ok": True, "version": None}))
    assert res["latest"] is None and res["newer"] is False


def test_check_unconfigured_raises(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.check()


# ------------------------------ classify --------------------------------

def test_classify_reboot_when_deps_unchanged(tmp_path):
    inst, stg = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); stg.mkdir()
    _write_pyproject(inst, ["click", "fastapi"])
    _write_pyproject(stg, ["click", "fastapi"])
    assert updater.classify(stg, inst) == "reboot"


def test_classify_deps_when_changed(tmp_path):
    inst, stg = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); stg.mkdir()
    _write_pyproject(inst, ["click"])
    _write_pyproject(stg, ["click", "httpx"])
    assert updater.classify(stg, inst) == "deps"


def test_classify_deps_when_extra_changes(tmp_path):
    inst, stg = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); stg.mkdir()
    _write_pyproject(inst, ["click"], extras={"voice": ["faster-whisper"]})
    _write_pyproject(stg, ["click"], extras={"voice": ["faster-whisper", "soundfile"]})
    assert updater.classify(stg, inst) == "deps"


def test_classify_setup_on_python_change(tmp_path):
    inst, stg = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); stg.mkdir()
    _write_pyproject(inst, ["click"], requires_python="==3.12.*")
    _write_pyproject(stg, ["click"], requires_python="==3.13.*")
    assert updater.classify(stg, inst) == "setup"


def test_classify_manifest_escalates_to_runtime(tmp_path):
    inst, stg = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); stg.mkdir()
    _write_pyproject(inst, ["click"])
    _write_pyproject(stg, ["click"])
    assert updater.classify(stg, inst, manifest={"needs": "runtime"}) == "runtime"


def test_classify_takes_max_of_manifest_and_auto(tmp_path):
    inst, stg = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); stg.mkdir()
    _write_pyproject(inst, ["click"])
    _write_pyproject(stg, ["click", "httpx"])         # auto-detect deps
    # manifest says only reboot, but deps detected -> deps wins (escalate, never down)
    assert updater.classify(stg, inst, manifest={"needs": "reboot"}) == "deps"


def test_read_manifest(tmp_path):
    (tmp_path / "update.json").write_text(
        json.dumps({"version": "0.2.0", "needs": "deps"}), encoding="utf-8")
    assert updater.read_manifest(tmp_path)["needs"] == "deps"
    assert updater.read_manifest(tmp_path / "nope") == {}


def test_class_summary_is_human():
    for k in ("reboot", "deps", "runtime", "setup"):
        assert isinstance(updater.class_summary(k), str) and updater.class_summary(k)


# ------------------------------ download --------------------------------

def test_download_streams_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "bugreport_upload_token": "tok"})
    dest = tmp_path / "build.zip"
    captured = {}

    def op(url, headers, timeout, d):
        captured["url"] = url
        captured["token"] = headers.get("X-Localm-Token")
        d.write_bytes(b"PKbuildzip")

    updater.download(7, dest, opener=op)
    assert dest.read_bytes() == b"PKbuildzip"
    assert "/update/download?id=7" in captured["url"]
    assert captured["token"] == "tok"


def test_download_bad_asset_id_raises(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.download("not-a-number", "/tmp/x")
