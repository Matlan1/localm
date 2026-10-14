# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/verify_release.py: checksum/signature verification for a downloaded
release asset. Reuses localm.updater.verify_signature (never reimplementing
the envelope check) against the same signing keys the auto-updater trusts, and
independently supports a plain SHA256 digest check."""

import hashlib
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import updater
from tests._update_signing import envelope, keyset

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import verify_release as vr  # noqa: E402


@pytest.fixture(autouse=True)
def _home(monkeypatch, tmp_path):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")


def _pin(monkeypatch):
    """Pin a fresh trust root; returns its release key."""
    release = Ed25519PrivateKey.generate()
    recovery = [Ed25519PrivateKey.generate() for _ in range(3)]
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(1, [release], recovery))
    return release


def _stranger():
    return Ed25519PrivateKey.generate()


def _sig(priv, data: bytes) -> str:
    return envelope(data, [priv])


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
    priv = _pin(monkeypatch)
    data = b"a release build"
    ok, msg = vr.verify_ed25519(data, _sig(priv, data))
    assert ok
    assert "update signing keys" in msg


def test_verify_ed25519_tampered_data_fails(monkeypatch):
    priv = _pin(monkeypatch)
    sig = _sig(priv, b"the original build")
    ok, msg = vr.verify_ed25519(b"a tampered build", sig)
    assert not ok
    assert "not signed by a trusted release key" in msg


def test_verify_ed25519_wrong_key_fails(monkeypatch):
    _pin(monkeypatch)
    priv = _stranger()
    data = b"build"
    ok, _msg = vr.verify_ed25519(data, _sig(priv, data))
    assert not ok


def test_verify_ed25519_malformed_signature_is_caught_not_raised(monkeypatch):
    _pin(monkeypatch)
    # Must return a (False, message) tuple; a LocalmError from verify_signature
    # (not an envelope) must never escape as a raw traceback.
    ok, msg = vr.verify_ed25519(b"build", "!!! not base64 !!!")
    assert not ok
    assert "0.2.0 or earlier" in msg


# ------------------------------------ main() --------------------------------

def test_main_signature_ok_exit_0(tmp_path, monkeypatch, capsys):
    priv = _pin(monkeypatch)
    data = b"a release zip's bytes"
    zpath = _write(tmp_path, "localm-0.1.5.zip", data)
    sig_path = _write(tmp_path, "localm-0.1.5.zip.sig", _sig(priv, data).encode())
    rc = vr.main([str(zpath), "--sig", str(sig_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "signature: OK" in out


def test_main_adjacent_sig_file_auto_discovered(tmp_path, monkeypatch):
    priv = _pin(monkeypatch)
    data = b"another release zip"
    zpath = _write(tmp_path, "localm-0.1.6.zip", data)
    _write(tmp_path, "localm-0.1.6.zip.sig", _sig(priv, data).encode())
    rc = vr.main([str(zpath)])   # no --sig flag: must auto-find <PATH>.sig
    assert rc == 0


def test_main_tampered_zip_exit_1(tmp_path, monkeypatch, capsys):
    priv = _pin(monkeypatch)
    good = b"the real bytes"
    sig_path = _write(tmp_path, "z.zip.sig", _sig(priv, good).encode())
    tampered = _write(tmp_path, "z.zip", b"tampered bytes, not the original")
    rc = vr.main([str(tampered), "--sig", str(sig_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "signature: FAILED" in out


def test_main_wrong_key_exit_1(tmp_path, monkeypatch):
    _pin(monkeypatch)
    priv = _stranger()
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
    _pin(monkeypatch)
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



def test_verify_ed25519_accepts_an_embedded_key_set_without_saving(monkeypatch, tmp_path):
    release = Ed25519PrivateKey.generate()
    a, b, c = (Ed25519PrivateKey.generate() for _ in range(3))
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(1, [release], [a, b, c]))
    new_release = Ed25519PrivateKey.generate()
    ks = keyset(2, [new_release], [a, b, c], signers=[a, b])
    data = b"a rotated release"
    ok, _msg = vr.verify_ed25519(data, envelope(data, [new_release], keyset=ks))
    assert ok
    assert not (tmp_path / "home").exists()
    ok, _msg = vr.verify_ed25519(data, envelope(data, [new_release]))
    assert not ok


def test_main_corrupt_saved_trust_is_could_not_check_exit_2(tmp_path, monkeypatch, capsys):
    priv = _pin(monkeypatch)
    trust = tmp_path / "home" / "updates" / "trust.json"
    trust.parent.mkdir(parents=True)
    trust.write_text("{broken", encoding="utf-8")
    data = b"build"
    zpath = _write(tmp_path, "z.zip", data)
    sig_path = _write(tmp_path, "z.zip.sig", _sig(priv, data).encode())
    assert vr.main([str(zpath), "--sig", str(sig_path)]) == 2
    err = capsys.readouterr().err
    assert "COULD NOT CHECK" in err and str(trust) in err
