# SPDX-License-Identifier: AGPL-3.0-or-later
"""Update signing trust across data dirs and a damaged trust file.

An accepted key set is saved in the data dir that applied the update, while the
install tree is shared by every data dir. A data dir without that file trusts the
key set pinned in the installed code, so a release that rotates keys pins the new
set in its own code too. A trust file that is not valid UTF-8 is refused as
corrupt on every path that loads it: ``localm update``, ``POST /api/update/apply``
and ``scripts/verify_release.py``.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from localm import _version, updater
from localm.bugreport import LocalmError
from tests._update_signing import Trust, envelope, keyset

NOT_UTF8 = b"\xff"


def _use_home(monkeypatch, home: Path) -> Path:
    monkeypatch.setenv("LOCALM_HOME", str(home))
    return home / "updates" / "trust.json"


def _build(version: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version)
        z.writestr("localm/__init__.py", f"# {version}")
        z.writestr("pyproject.toml",
                   '[project]\nname = "localm"\ndependencies = ["click"]\n')
    return buf.getvalue()


def _install(tmp_path: Path) -> Path:
    inst = tmp_path / "inst"
    (inst / "localm").mkdir(parents=True)
    (inst / "localm" / "__init__.py").write_text("# 0.1.0", encoding="utf-8")
    (inst / "VERSION").write_text("0.1.0", encoding="utf-8")
    (inst / "pyproject.toml").write_text(
        '[project]\nname = "localm"\ndependencies = ["click"]\n', encoding="utf-8")
    return inst


def _opener(data: bytes):
    def dl(url, headers, timeout, dest):
        Path(dest).write_bytes(data)
    return dl


def _apply(inst, data, sig):
    return updater.apply(5, installed=inst, signature=sig,
                         download_opener=_opener(data), runner=lambda c: 0)


def _installed_version(inst: Path) -> str:
    return (inst / "VERSION").read_text(encoding="utf-8").strip()


@pytest.fixture
def apply_env(monkeypatch):
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")


# ------------------- a rotation and a second data dir ----------------------

def test_second_data_dir_refuses_a_revoked_key_once_the_code_pins_the_rotation(
        tmp_path, monkeypatch, apply_env):
    trust = Trust(monkeypatch)
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    rotation = keyset(2, [r2], trust.recovery, signers=[a, b])
    inst = _install(tmp_path)

    trust_a = _use_home(monkeypatch, tmp_path / "home-a")
    first = _build("0.2.0")
    _apply(inst, first, envelope(first, [r2], keyset=rotation))
    assert _installed_version(inst) == "0.2.0"
    assert json.loads(trust_a.read_text(encoding="utf-8")) == rotation

    monkeypatch.setattr(updater, "_PINNED_KEYSET",
                        {k: rotation[k] for k in ("seq", "release", "recovery")})
    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    trust_b = _use_home(monkeypatch, tmp_path / "home-b")
    forged = _build("0.3.0")
    with pytest.raises(LocalmError) as ei:
        _apply(inst, forged, envelope(forged, [trust.r]))
    assert "not signed by a trusted release key" in ei.value.reason
    assert _installed_version(inst) == "0.2.0"
    assert not trust_b.exists()

    _apply(inst, forged, envelope(forged, [r2]))
    assert _installed_version(inst) == "0.3.0"


def test_second_data_dir_accepts_a_revoked_key_when_the_code_still_pins_the_old_set(
        tmp_path, monkeypatch, apply_env):
    trust = Trust(monkeypatch)
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    rotation = keyset(2, [r2], trust.recovery, signers=[a, b])
    inst = _install(tmp_path)

    _use_home(monkeypatch, tmp_path / "home-a")
    first = _build("0.2.0")
    _apply(inst, first, envelope(first, [r2], keyset=rotation))

    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    _use_home(monkeypatch, tmp_path / "home-b")
    forged = _build("0.3.0")
    _apply(inst, forged, envelope(forged, [trust.r]))
    assert _installed_version(inst) == "0.3.0"


def test_pinned_set_newer_than_the_saved_one_wins(tmp_path, monkeypatch):
    trust = Trust(monkeypatch)
    a, b, _c = trust.recovery
    r2, r3 = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    f = _use_home(monkeypatch, tmp_path / "home")
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(keyset(2, [r2], trust.recovery, signers=[a, b])),
                 encoding="utf-8")
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(3, [r3], trust.recovery))
    data = b"build"
    assert updater.verify_signature(data, envelope(data, [r3])) is None
    with pytest.raises(LocalmError):
        updater.verify_signature(data, envelope(data, [r2]))


# ------------------------ trust file not UTF-8 ----------------------------

@pytest.fixture
def broken_trust(tmp_path, monkeypatch):
    trust = Trust(monkeypatch)
    f = _use_home(monkeypatch, tmp_path / "home")
    f.parent.mkdir(parents=True)
    f.write_bytes(NOT_UTF8)
    return trust, f


def test_load_trusted_refuses_a_non_utf8_trust_file_as_corrupt(broken_trust):
    _trust, f = broken_trust
    with pytest.raises(LocalmError) as ei:
        updater._load_trusted()
    assert "corrupt" in ei.value.summary and str(f) in ei.value.reason
    assert f.read_bytes() == NOT_UTF8


def _check_reply(trust, data):
    return {"current": "0.1.0", "latest": "0.2.0", "newer": True, "comparable": True,
            "notes": None, "asset": {"id": 7},
            "signature": envelope(data, [trust.r])}


def _real_apply_with(monkeypatch, tmp_path, data):
    inst = _install(tmp_path)
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "repo_root", lambda: inst)
    monkeypatch.setattr(updater, "is_site_packages_install", lambda t: False)
    monkeypatch.setattr(updater, "download",
                        lambda aid, dest, **kw: Path(dest).write_bytes(data) and Path(dest))
    return inst


def test_cli_update_reports_a_non_utf8_trust_file(broken_trust, cli_runner, monkeypatch,
                                                   tmp_path):
    from localm.cli.maintenance import update_cmd
    trust, f = broken_trust
    data = _build("0.2.0")
    inst = _real_apply_with(monkeypatch, tmp_path, data)
    monkeypatch.setattr(updater, "check", lambda **_: _check_reply(trust, data))
    monkeypatch.setenv("LOCALM_HOME", str(f.parents[1]))
    r = cli_runner.invoke(update_cmd, ["--yes"])
    assert r.exit_code == 0, r.output
    assert "Update failed" in r.output and "corrupt" in r.output, r.output
    assert "unexpected error" not in r.output, r.output
    assert _installed_version(inst) == "0.1.0"


def test_api_update_apply_reports_a_non_utf8_trust_file(broken_trust, monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from localm.inference.http_server import create_app
    trust, _f = broken_trust
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    data = _build("0.2.0")
    inst = _real_apply_with(monkeypatch, tmp_path, data)
    monkeypatch.setattr(updater, "check", lambda: _check_reply(trust, data))
    engine = MagicMock()
    engine.display_name = "m"
    engine.loaded = True
    app = create_app(engine)
    with TestClient(app) as c:
        resp = c.post("/api/update/apply",
                      headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is False and "corrupt" in body["error"], body
    assert _installed_version(inst) == "0.1.0"


def test_verify_release_reports_a_non_utf8_trust_file_as_could_not_check(
        broken_trust, tmp_path, capsys):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import verify_release as vr
    trust, f = broken_trust
    data = b"build"
    zpath = tmp_path / "z.zip"
    zpath.write_bytes(data)
    sig = tmp_path / "z.zip.sig"
    sig.write_text(envelope(data, [trust.r]), encoding="utf-8")
    assert vr.main([str(zpath), "--sig", str(sig)]) == 2
    err = capsys.readouterr().err
    assert "COULD NOT CHECK" in err and "corrupt" in err and str(f) in err
