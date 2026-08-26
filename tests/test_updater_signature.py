# SPDX-License-Identifier: AGPL-3.0-or-later
"""The self-updater verifies an Ed25519 signature over the downloaded build
against a PINNED public key before extracting or executing it. Signing is
auth-model style: with a key pinned it ENFORCES (a missing, unsigned, tampered,
or downgraded build all refuse before any swap); with NO key pinned it fails OPEN
(applies on the HTTPS + private-channel trust) so the feature is not inert out of
the box. The SHIPPED default pins a key, and a guard below keeps it non-empty so
an empty key cannot brick self-update.
"""

import base64
import io
import zipfile

import pytest
from cryptography.hazmat.primitives import serialization as _ser
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import _version, updater
from localm.bugreport import LocalmError


def _keypair():
    priv = Ed25519PrivateKey.generate()
    pub_hex = priv.public_key().public_bytes(
        encoding=_ser.Encoding.Raw, format=_ser.PublicFormat.Raw).hex()
    return priv, pub_hex


def _sig(priv, data: bytes) -> str:
    return base64.b64encode(priv.sign(data)).decode("ascii")


# ------------------------- verify_signature -----------------------------

def test_valid_signature_passes(monkeypatch):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    data = b"a localm build.zip"
    updater.verify_signature(data, _sig(priv, data))   # must not raise


def test_tampered_data_fails(monkeypatch):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    sig = _sig(priv, b"the original build")
    with pytest.raises(LocalmError):
        updater.verify_signature(b"a tampered build", sig)


def test_wrong_key_fails(monkeypatch):
    priv, _ = _keypair()
    _, other_pub = _keypair()                    # pin a DIFFERENT key
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (other_pub,))
    data = b"build"
    with pytest.raises(LocalmError):
        updater.verify_signature(data, _sig(priv, data))


def test_missing_signature_fails_closed(monkeypatch):
    _, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    with pytest.raises(LocalmError):
        updater.verify_signature(b"build", None)


def test_no_pinned_key_allows_unsigned(monkeypatch):
    # Fail-OPEN when unconfigured (no key pinned): verification allows the
    # update rather than bricking it. The shipped default DOES pin a key.
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", ())
    updater.verify_signature(b"build", None)                # must NOT raise
    updater.verify_signature(b"build", "c2ln")              # a stray sig is ignored too


def test_shipped_pubkeys_are_pinned(monkeypatch):
    # The real shipped tuple must stay non-empty and every pin must be a valid
    # 32-byte (64-hex) Ed25519 public key.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    assert updater._UPDATE_PUBKEYS, "_UPDATE_PUBKEYS must not ship empty (see updater.py)"
    for hexkey in updater._UPDATE_PUBKEYS:
        raw = bytes.fromhex(str(hexkey).strip())
        assert len(raw) == 32, f"pin is not 32 bytes: {hexkey!r}"
        Ed25519PublicKey.from_public_bytes(raw)             # raises if not a valid key


def test_malformed_signature_fails(monkeypatch):
    _, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    with pytest.raises(LocalmError):
        updater.verify_signature(b"build", "!!! not base64 !!!")


def test_key_rotation_allowlist(monkeypatch):
    # Two pinned keys (rotation window): a build signed by EITHER validates.
    old, old_pub = _keypair()
    new, new_pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (old_pub, new_pub))
    data = b"build"
    updater.verify_signature(data, _sig(old, data))
    updater.verify_signature(data, _sig(new, data))


def test_all_pins_malformed_names_the_real_cause(monkeypatch, caplog):
    # With a key PINNED but unparseable, the refusal names the broken pin rather
    # than reporting that verification is not set up, and the skipped pin is
    # warned about.
    import logging
    priv, _ = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", ("not-hex",))
    with caplog.at_level(logging.WARNING, logger="localm"):
        with pytest.raises(LocalmError) as ei:
            updater.verify_signature(b"build", _sig(priv, b"build"))
    assert "no pinned signing key is usable" in ei.value.summary
    assert "none parses" in ei.value.reason
    assert "does not parse as an Ed25519" in caplog.text


# --------------------------- anti-rollback ------------------------------

def test_refuse_downgrade(monkeypatch):
    monkeypatch.setattr(_version, "read_version", lambda: "0.5.0")
    updater._refuse_downgrade("0.6.0")                 # newer -> ok
    with pytest.raises(LocalmError):
        updater._refuse_downgrade("0.5.0")             # equal -> refuse
    with pytest.raises(LocalmError):
        updater._refuse_downgrade("0.4.0")             # older -> refuse
    with pytest.raises(LocalmError):
        updater._refuse_downgrade("")                  # no version -> refuse


# ------------------ apply(): refuse BEFORE any swap ---------------------

def _build_zip_bytes(version="0.2.0", deps=("click",)) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version)
        z.writestr("pyproject.toml", '[project]\nname = "localm"\ndependencies = ['
                   + ", ".join(f'"{d}"' for d in deps) + ']\n')
        z.writestr("localm/__init__.py", f"# {version}")
    return buf.getvalue()


def _opener_for(data: bytes):
    def dl(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return dl


def _fake_install(tmp_path):
    inst = tmp_path / "inst"
    (inst / "localm").mkdir(parents=True)
    (inst / "localm" / "__init__.py").write_text("# 0.1.0", encoding="utf-8")
    (inst / "VERSION").write_text("0.1.0", encoding="utf-8")
    (inst / "pyproject.toml").write_text(
        '[project]\nname = "localm"\ndependencies = ["click"]\n', encoding="utf-8")
    return inst


def _assert_untouched(inst):
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    assert (inst / "localm" / "__init__.py").read_text() == "# 0.1.0"


@pytest.fixture
def apply_cfg(monkeypatch, tmp_path):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")


def test_apply_refuses_unsigned_build_before_swap(tmp_path, monkeypatch, apply_cfg):
    _, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    inst = _fake_install(tmp_path)
    with pytest.raises(LocalmError):
        updater.apply(5, installed=inst, signature=None,
                      download_opener=_opener_for(_build_zip_bytes("0.2.0")))
    _assert_untouched(inst)


def test_apply_applies_when_no_key_pinned(tmp_path, monkeypatch, apply_cfg):
    # Fail-OPEN: with no key pinned the signature stage does not block, so an
    # update applies out of the box (a reboot-class swap here).
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", ())
    inst = _fake_install(tmp_path)
    data = _build_zip_bytes("0.2.0")
    res = updater.apply(5, installed=inst, signature=None,
                        download_opener=_opener_for(data), runner=lambda c: 0)
    assert res["applied"] and res["version"] == "0.2.0"
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    assert (inst / "localm" / "__init__.py").read_text() == "# 0.2.0"


def test_apply_refuses_tampered_build_before_swap(tmp_path, monkeypatch, apply_cfg):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    inst = _fake_install(tmp_path)
    good = _build_zip_bytes("0.2.0", deps=("click",))
    sig = _sig(priv, good)
    tampered = _build_zip_bytes("0.2.0", deps=("click", "evil"))   # different bytes
    with pytest.raises(LocalmError):
        updater.apply(5, installed=inst, signature=sig,
                      download_opener=_opener_for(tampered))
    _assert_untouched(inst)


def test_apply_refuses_downgrade_before_swap(tmp_path, monkeypatch, apply_cfg):
    priv, pub = _keypair()
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub,))
    monkeypatch.setattr(_version, "read_version", lambda: "0.9.0")   # newer than the build
    inst = _fake_install(tmp_path)
    old_build = _build_zip_bytes("0.2.0")                # a VALIDLY signed but OLD build
    with pytest.raises(LocalmError):
        updater.apply(5, installed=inst, signature=_sig(priv, old_build),
                      download_opener=_opener_for(old_build))
    _assert_untouched(inst)


# ---------------- end-to-end: the maintainer's signer -------------------

def test_sign_release_script_roundtrip(tmp_path, monkeypatch):
    """scripts/sign_release.py generates a key + signs a build, and the updater
    verifies it against the derived public key - the whole maintainer flow."""
    import importlib.util
    from pathlib import Path
    script = Path(updater.__file__).resolve().parents[1] / "scripts" / "sign_release.py"
    if not script.is_file():
        # scripts/sign_release.py is gitignored, maintainer-only release
        # tooling, so a fresh clone or an uncopied worktree does not have this
        # file. The import runs while the test executes, so a plain pytest.skip
        # is enough; the rest of the file keeps running.
        pytest.skip(f"{script} not present (gitignored maintainer-only "
                    "release tooling, AGENTS.md rule 6) - skipping the "
                    "sign_release.py round-trip")
    spec = importlib.util.spec_from_file_location("sign_release", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # a dummy tmp_path key's real ACL is environment-dependent, so isolate it
    # like every other gate in this orchestration test.
    monkeypatch.setattr(mod, "check_key_permissions", lambda *a, **k: None)

    keyfile = tmp_path / "signing_key.pem"
    assert mod._gen(keyfile) == 0 and keyfile.is_file()
    priv = _ser.load_pem_private_key(keyfile.read_bytes(), password=None)
    pub_hex = priv.public_key().public_bytes(
        _ser.Encoding.Raw, _ser.PublicFormat.Raw).hex()

    build = tmp_path / "build.zip"
    build.write_bytes(_build_zip_bytes("0.2.0"))
    assert mod._sign(build, keyfile, None) == 0
    sig = (tmp_path / "build.zip.sig").read_text().strip()

    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (pub_hex,))
    updater.verify_signature(build.read_bytes(), sig)   # must not raise
