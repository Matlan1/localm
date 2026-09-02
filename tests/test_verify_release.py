# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/verify_release.py: checksum/signature verification for a downloaded
release asset. Reuses localm.updater.verify_signature (never reimplementing
key loading) against the same pinned Ed25519 key the auto-updater trusts, and
independently supports a plain SHA256 digest check."""

import base64
import hashlib
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization as _ser
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import updater

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import verify_release as vr  # noqa: E402


def _keypair():
    priv = Ed25519PrivateKey.generate()
    pub_hex = priv.public_key().public_bytes(
        encoding=_ser.Encoding.Raw, format=_ser.PublicFormat.Raw).hex()
    return priv, pub_hex


def _sig(priv, data: bytes) -> str:
    return base64.b64encode(priv.sign(data)).decode("ascii")


def _write(tmp_path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# --------------------------- sha256_of / parse_sha256_arg ------------------

def test_sha256_of_matches_hashlib(tmp_path):
    data = b"some release bytes"
    p = _write(tmp_path, "r.zip", data)
    assert vr.sha256_of(p) == hashlib.sha256(data).hexdigest()


def test_parse_sha256_arg_bare_hex():
    hexdigest = "a" * 64
    assert vr.parse_sha256_arg(hexdigest) == hexdigest
    assert vr.parse_sha256_arg(hexdigest.upper()) == hexdigest


def test_parse_sha256_arg_plain_hex_file(tmp_path):
    hexdigest = "b" * 64
    p = tmp_path / "digest.txt"
    p.write_text(hexdigest + "\n")
    assert vr.parse_sha256_arg(str(p)) == hexdigest


def test_parse_sha256_arg_two_column_sha256sum_file(tmp_path):
    hexdigest = "c" * 64
    p = tmp_path / "digest.sha256"
    p.write_text(f"{hexdigest}  localm-0.1.5.zip\n")
    assert vr.parse_sha256_arg(str(p)) == hexdigest


def test_parse_sha256_arg_bad_value_raises():
    with pytest.raises(ValueError):
        vr.parse_sha256_arg("not-a-digest-and-not-a-file")


def test_import_error_message_is_actionable():
    msg = vr._import_error_message(ModuleNotFoundError("No module named 'cryptography'"))
    assert "cryptography" in msg
    assert "pip install" in msg


# --------------------------------- verify_ed25519 ---------------------------

def test_verify_ed25519_valid_signature(monkeypatch):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    data = b"a release build"
    ok, msg = vr.verify_ed25519(data, _sig(priv, data))
    assert ok
    assert "pinned update key" in msg


def test_verify_ed25519_tampered_data_fails(monkeypatch):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    sig = _sig(priv, b"the original build")
    ok, msg = vr.verify_ed25519(b"a tampered build", sig)
    assert not ok
    assert "did not match" in msg


def test_verify_ed25519_wrong_key_fails(monkeypatch):
    priv, _ = _keypair()
    _, other_pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (other_pub,))
    data = b"build"
    ok, _msg = vr.verify_ed25519(data, _sig(priv, data))
    assert not ok


def test_verify_ed25519_malformed_signature_is_caught_not_raised(monkeypatch):
    _, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    # Must return a (False, message) tuple; a LocalmError from verify_signature
    # (malformed base64) must never escape as a raw traceback.
    ok, msg = vr.verify_ed25519(b"build", "!!! not base64 !!!")
    assert not ok
    assert "malformed" in msg


# ------------------------------------ main() --------------------------------

def test_main_signature_ok_exit_0(tmp_path, monkeypatch, capsys):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    data = b"a release zip's bytes"
    zpath = _write(tmp_path, "localm-0.1.5.zip", data)
    sig_path = _write(tmp_path, "localm-0.1.5.zip.sig", _sig(priv, data).encode())
    rc = vr.main([str(zpath), "--sig", str(sig_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "signature: OK" in out


def test_main_adjacent_sig_file_auto_discovered(tmp_path, monkeypatch):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    data = b"another release zip"
    zpath = _write(tmp_path, "localm-0.1.6.zip", data)
    _write(tmp_path, "localm-0.1.6.zip.sig", _sig(priv, data).encode())
    rc = vr.main([str(zpath)])   # no --sig flag: must auto-find <PATH>.sig
    assert rc == 0


def test_main_tampered_zip_exit_1(tmp_path, monkeypatch, capsys):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    good = b"the real bytes"
    sig_path = _write(tmp_path, "z.zip.sig", _sig(priv, good).encode())
    tampered = _write(tmp_path, "z.zip", b"tampered bytes, not the original")
    rc = vr.main([str(tampered), "--sig", str(sig_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "signature: FAILED" in out


def test_main_wrong_key_exit_1(tmp_path, monkeypatch):
    priv, _pub = _keypair()
    _, other_pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (other_pub,))
    data = b"build content"
    zpath = _write(tmp_path, "z.zip", data)
    sig_path = _write(tmp_path, "z.zip.sig", _sig(priv, data).encode())
    rc = vr.main([str(zpath), "--sig", str(sig_path)])
    assert rc == 1


def test_main_sha256_bare_hex_match_exit_0(tmp_path):
    data = b"content for sha256 check"
    zpath = _write(tmp_path, "z.zip", data)
    digest = hashlib.sha256(data).hexdigest()
    rc = vr.main([str(zpath), "--sha256", digest])
    assert rc == 0


def test_main_sha256_bare_hex_mismatch_exit_1(tmp_path, capsys):
    data = b"content for sha256 check"
    zpath = _write(tmp_path, "z.zip", data)
    rc = vr.main([str(zpath), "--sha256", "f" * 64])
    assert rc == 1
    out = capsys.readouterr().out
    assert "does NOT match" in out


def test_main_sha256_two_column_file_match_exit_0(tmp_path):
    data = b"checksum file content test"
    zpath = _write(tmp_path, "localm-1.0.0.zip", data)
    digest = hashlib.sha256(data).hexdigest()
    digest_file = _write(tmp_path, "localm-1.0.0.zip.sha256",
                         f"{digest}  localm-1.0.0.zip\n".encode())
    rc = vr.main([str(zpath), "--sha256", str(digest_file)])
    assert rc == 0


def test_main_sha256_two_column_file_mismatch_exit_1(tmp_path):
    data = b"checksum file content test 2"
    zpath = _write(tmp_path, "localm-1.0.1.zip", data)
    digest_file = _write(tmp_path, "localm-1.0.1.zip.sha256",
                         f"{'0' * 64}  localm-1.0.1.zip\n".encode())
    rc = vr.main([str(zpath), "--sha256", str(digest_file)])
    assert rc == 1


def test_main_no_sig_no_sha256_no_adjacent_file_exit_2(tmp_path):
    zpath = _write(tmp_path, "lonely.zip", b"nothing to check against")
    rc = vr.main([str(zpath)])
    assert rc == 2


def test_main_missing_sig_file_exit_2(tmp_path):
    zpath = _write(tmp_path, "z.zip", b"data")
    rc = vr.main([str(zpath), "--sig", str(tmp_path / "does-not-exist.sig")])
    assert rc == 2


def test_main_missing_zip_file_exit_2(tmp_path):
    rc = vr.main([str(tmp_path / "does-not-exist.zip")])
    assert rc == 2


def test_main_sha256_only_does_not_auto_check_adjacent_sig(tmp_path):
    # --sha256 given with no --sig: an adjacent .sig file must NOT be
    # auto-discovered (auto-discovery only applies when NEITHER flag is
    # given at all), so this succeeds on the sha256 match alone even though
    # a (deliberately invalid) .sig sits right next to it.
    data = b"only sha256 was requested"
    zpath = _write(tmp_path, "z.zip", data)
    _write(tmp_path, "z.zip.sig", b"not a valid signature at all")
    digest = hashlib.sha256(data).hexdigest()
    rc = vr.main([str(zpath), "--sha256", digest])
    assert rc == 0


def test_main_localm_error_from_malformed_signature_reported_not_raised(tmp_path, monkeypatch, capsys):
    _, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    zpath = _write(tmp_path, "z.zip", b"data")
    sig_path = _write(tmp_path, "z.zip.sig", b"!!! not base64 !!!")
    rc = vr.main([str(zpath), "--sig", str(sig_path)])   # must not raise
    assert rc == 1
    out = capsys.readouterr().out
    assert "signature: FAILED" in out
    assert "malformed" in out


def test_main_signature_import_error_is_could_not_check_not_a_crash(tmp_path, monkeypatch):
    # Simulates cryptography/rich not being installed (a bare, not-yet-set-up
    # clone) without actually breaking the test environment's own imports.
    def _raise(*_a, **_k):
        raise ImportError("No module named 'cryptography'")
    monkeypatch.setattr(vr, "verify_ed25519", _raise)
    zpath = _write(tmp_path, "z.zip", b"data")
    sig_path = _write(tmp_path, "z.zip.sig", b"c2ln")
    rc = vr.main([str(zpath), "--sig", str(sig_path)])   # must not raise/crash
    assert rc == 2
