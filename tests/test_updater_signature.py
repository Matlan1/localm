# SPDX-License-Identifier: AGPL-3.0-or-later
"""The self-updater verifies a JSON signature envelope over the downloaded build
before extracting or executing it. Builds are signed by the trusted release key
set; a replacement key set is accepted only with a higher seq and signatures from
the trusted recovery threshold of trusted recovery keys, and is saved only after
the whole update verifies. Every key here is a real Ed25519 key made in the test.
"""

import io
import json
import zipfile

import pytest
from cryptography.hazmat.primitives import serialization as _ser
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import _version, updater
from localm.bugreport import LocalmError
from tests._update_signing import Trust, envelope, keypair, keyset, pub_hex, sig_b64

DATA = b"a localm build.zip"


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: h)
    return h


@pytest.fixture
def trust(monkeypatch, home):
    return Trust(monkeypatch)


def _trust_file(home):
    return home / "updates" / "trust.json"


def _refused(data, sig, **kw):
    with pytest.raises(LocalmError) as ei:
        updater.verify_signature(data, sig, **kw)
    return ei.value


# ----------------------------- shipped trust ------------------------------

def test_shipped_keyset_pins_the_release_and_recovery_keys():
    ks = updater._parse_keyset(updater._PINNED_KEYSET, signed=False)
    assert ks["seq"] == 1
    assert ks["release"] == {"threshold": 1, "keys": [
        "4faca1439ff49ca9269580061f17d1847e079c35c82e8372b221bc3abba6db40"]}
    assert ks["recovery"] == {"threshold": 2, "keys": [
        "d5357ae966ce0eecc687f40ca3a68b93c4d8e1c246b9af7006ee03905f5507ee",
        "090193021f42f8514b60676b38c0365034e3bf312d5f4946e1434e328f4a78e8",
        "1a9a3fc808a1e2ab207b71fd4b6eeb09860b5ef17281d6f13c73dcf66f73f334"]}


def test_single_key_pin_is_gone():
    assert not hasattr(updater, "_UPDATE_PUBKEYS")
    assert not hasattr(updater, "_load_update_pubkeys")


def test_canonical_keyset_bytes_format():
    ks = {"seq": 2, "release": {"threshold": 1, "keys": ["bb"]},
          "recovery": {"threshold": 2, "keys": ["cc", "aa"]},
          "signatures": [{"key": "x", "sig": "y"}]}
    assert updater.canonical_keyset_bytes(ks) == (
        b'{"recovery":{"keys":["cc","aa"],"threshold":2},'
        b'"release":{"keys":["bb"],"threshold":1},"seq":2}')


# ------------------------------ build signatures ---------------------------

def test_release_signed_build_is_accepted(trust):
    assert updater.verify_signature(DATA, envelope(DATA, [trust.r])) is None


def test_envelope_with_trailing_newline_is_accepted(trust):
    assert updater.verify_signature(DATA, envelope(DATA, [trust.r]) + "\n") is None


def test_recovery_signed_build_is_refused(trust):
    e = _refused(DATA, envelope(DATA, trust.recovery))
    assert "not signed by a trusted release key" in e.reason


def test_unknown_key_is_refused(trust):
    stranger = Ed25519PrivateKey.generate()
    assert "not signed by a trusted release key" in _refused(
        DATA, envelope(DATA, [stranger])).reason


def test_tampered_build_is_refused(trust):
    assert "not signed by a trusted release key" in _refused(
        b"a tampered build", envelope(DATA, [trust.r])).reason


def test_signature_by_trusted_key_claimed_for_another_key_is_refused(trust):
    stranger = Ed25519PrivateKey.generate()
    env = {"format": updater._SIG_FORMAT,
           "signatures": [{"key": pub_hex(trust.r), "sig": sig_b64(stranger, DATA)}]}
    _refused(DATA, json.dumps(env))


def test_repeated_signature_counts_once(trust, monkeypatch):
    r2 = Ed25519PrivateKey.generate()
    pinned = keyset(1, [trust.r, r2], trust.recovery, release_threshold=2)
    monkeypatch.setattr(updater, "_PINNED_KEYSET", pinned)
    _refused(DATA, envelope(DATA, [trust.r, trust.r]))
    assert updater.verify_signature(DATA, envelope(DATA, [trust.r, r2])) is None


def test_bad_signature_does_not_mask_a_good_one_from_the_same_key(trust):
    stranger = Ed25519PrivateKey.generate()
    env = json.loads(envelope(DATA, [trust.r]))
    env["signatures"].insert(0, {"key": pub_hex(trust.r), "sig": sig_b64(stranger, DATA)})
    assert updater.verify_signature(DATA, json.dumps(env)) is None


@pytest.mark.parametrize("sig", [None, "", "   ", 5, b"bytes", ["x"]])
def test_missing_signature_is_refused(trust, sig):
    assert "unsigned" in _refused(DATA, sig).summary


def test_legacy_bare_base64_signature_is_refused(trust):
    e = _refused(DATA, sig_b64(trust.r, DATA))
    assert "0.2.0 or earlier" in e.reason


def test_oversized_signature_is_refused_before_parsing(trust, monkeypatch):
    def boom(_):
        raise AssertionError("parsed an oversized envelope")
    text = envelope(DATA, [trust.r])
    text += " " * (updater._SIG_MAX_CHARS + 1 - len(text))
    monkeypatch.setattr(updater, "_strict_json", boom)
    assert "longer than 4096" in _refused(DATA, text).reason


def _env_obj(trust):
    return json.loads(envelope(DATA, [trust.r]))


@pytest.mark.parametrize("mutate,needle", [
    (lambda e: "[]", "not an object"),
    (lambda e: "{" * 3000, "is not a localm-update-sig/2"),
    (lambda e: "[" * 4000, "is not a localm-update-sig/2"),
    (lambda e: '{"format": "localm-update-sig/2", "format": "x", "signatures": []}',
     "duplicate field"),
    (lambda e: {**e, "format": "localm-update-sig/1"}, "unsupported signature format"),
    (lambda e: {k: v for k, v in e.items() if k != "format"}, "lacks format"),
    (lambda e: {k: v for k, v in e.items() if k != "signatures"}, "lacks signatures"),
    (lambda e: {**e, "extra": 1}, "unknown field(s) extra"),
    (lambda e: {**e, "signatures": {"key": "a"}}, "is not a list"),
    (lambda e: {**e, "signatures": ["x"]}, "not an object"),
    (lambda e: {**e, "signatures": [{"key": 1, "sig": "x"}]}, "pair of strings"),
    (lambda e: {**e, "signatures": [{**e["signatures"][0], "sig": "!!!"}]},
     "not valid base64"),
    (lambda e: {**e, "signatures": [{**e["signatures"][0], "sig": "c2ln"}]},
     "not 64 bytes"),
    (lambda e: {**e, "signatures": [{**e["signatures"][0], "x": 1}]}, "unknown field"),
])
def test_malformed_envelope_is_refused(trust, mutate, needle):
    bad = mutate(_env_obj(trust))
    text = bad if isinstance(bad, str) else json.dumps(bad)
    assert needle in _refused(DATA, text).reason


def test_nan_in_envelope_is_refused(trust):
    text = envelope(DATA, [trust.r])[:-1] + ', "keyset": NaN}'
    assert "is not a localm-update-sig/2" in _refused(DATA, text).reason


# --------------------------- key set replacement ---------------------------

def test_keyset_signed_by_two_recovery_keys_replaces_the_release_key(trust):
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, b])
    accepted = updater.verify_signature(DATA, envelope(DATA, [r2], keyset=ks))
    assert accepted["seq"] == 2 and accepted["release"]["keys"] == [pub_hex(r2)]
    assert accepted["signatures"] == ks["signatures"]
    # The old release key cannot sign a build that carries the new key set.
    _refused(DATA, envelope(DATA, [trust.r], keyset=ks))


def test_keyset_signed_by_one_recovery_key_is_refused(trust):
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[trust.recovery[0]])
    e = _refused(DATA, envelope(DATA, [r2], keyset=ks))
    assert "not signed by 2 trusted recovery keys" in e.reason


def test_keyset_signed_twice_by_one_recovery_key_is_refused(trust):
    a = trust.recovery[0]
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, a])
    _refused(DATA, envelope(DATA, [r2], keyset=ks))


def test_keyset_signed_by_a_non_current_recovery_key_is_refused(trust):
    a = trust.recovery[0]
    d = Ed25519PrivateKey.generate()          # in the NEW recovery set only
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], [a, d, trust.recovery[2]], signers=[a, d])
    _refused(DATA, envelope(DATA, [r2], keyset=ks))


def test_keyset_signed_by_release_key_is_refused(trust):
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[trust.r, trust.recovery[0]])
    _refused(DATA, envelope(DATA, [r2], keyset=ks))


def test_keyset_with_tampered_body_is_refused(trust):
    a, b, _c = trust.recovery
    r2, evil = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, b])
    ks["release"]["keys"] = [pub_hex(evil)]
    _refused(DATA, envelope(DATA, [evil], keyset=ks))


def test_lower_seq_keyset_is_refused(trust, monkeypatch):
    a, b, _c = trust.recovery
    monkeypatch.setattr(updater, "_PINNED_KEYSET",
                        keyset(3, [trust.r], trust.recovery))
    old = keyset(2, [trust.r], trust.recovery, signers=[a, b])
    assert "older than the trusted key set 3" in _refused(
        DATA, envelope(DATA, [trust.r], keyset=old)).reason


def test_equal_seq_identical_keyset_is_a_no_op(trust):
    a, b, _c = trust.recovery
    same = keyset(1, [trust.r], trust.recovery, signers=[a, b])
    assert updater.verify_signature(DATA, envelope(DATA, [trust.r], keyset=same)) is None


def test_equal_seq_different_keyset_is_refused(trust):
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    other = keyset(1, [r2], trust.recovery, signers=[a, b])
    assert "different key set 1" in _refused(
        DATA, envelope(DATA, [r2], keyset=other)).reason


@pytest.mark.parametrize("kw,needle", [
    ({"recovery_threshold": 1}, "recovery set threshold must be an integer of at least 2"),
    ({"release_threshold": 0}, "release set threshold must be an integer of at least 1"),
    ({"recovery_threshold": 4}, "recovery set has fewer keys than its threshold"),
])
def test_keyset_thresholds_are_enforced(trust, kw, needle):
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, b], **kw)
    assert needle in _refused(DATA, envelope(DATA, [r2], keyset=ks)).reason


@pytest.mark.parametrize("mutate,needle", [
    (lambda ks: ks["recovery"]["keys"].append(ks["recovery"]["keys"][0]), "not distinct"),
    (lambda ks: ks["release"]["keys"].append(ks["recovery"]["keys"][0]),
     "in both the release and the recovery set"),
    (lambda ks: ks["release"]["keys"].__setitem__(0, "AB" * 32), "lowercase hex"),
    (lambda ks: ks["release"]["keys"].__setitem__(0, "ab"), "lowercase hex"),
    (lambda ks: ks.__setitem__("seq", True), "seq must be a positive integer"),
    (lambda ks: ks.__setitem__("seq", 2.0), "seq must be a positive integer"),
    (lambda ks: ks.__setitem__("seq", 0), "seq must be a positive integer"),
    (lambda ks: ks["release"].__setitem__("threshold", True), "threshold must be"),
    (lambda ks: ks["release"].__setitem__("keys", []), "non-empty list"),
    (lambda ks: ks.pop("signatures"), "lacks signatures"),
    (lambda ks: ks.__setitem__("extra", 1), "unknown field(s) extra"),
])
def test_malformed_keyset_is_refused(trust, mutate, needle):
    a, b, _c = trust.recovery
    ks = keyset(2, [Ed25519PrivateKey.generate()], trust.recovery, signers=[a, b])
    mutate(ks)
    assert needle in _refused(DATA, envelope(DATA, [trust.r], keyset=ks)).reason


# ---------------------------- persisted trust ------------------------------

def test_corrupt_saved_keyset_is_refused_and_named(trust, home):
    f = _trust_file(home)
    f.parent.mkdir(parents=True)
    f.write_text("{not json", encoding="utf-8")
    e = _refused(DATA, envelope(DATA, [trust.r]))
    assert "corrupt" in e.summary and str(f) in e.reason


def test_invalid_saved_keyset_is_refused_not_replaced_by_pinned(trust, home):
    a, b, _c = trust.recovery
    ks = keyset(2, [trust.r], trust.recovery, signers=[a, b], recovery_threshold=1)
    f = _trust_file(home)
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(ks), encoding="utf-8")
    e = _refused(DATA, envelope(DATA, [trust.r]))
    assert str(f) in e.reason
    assert f.read_text(encoding="utf-8") == json.dumps(ks)


def test_unreadable_saved_keyset_is_refused_and_named(trust, home):
    f = _trust_file(home)
    f.mkdir(parents=True)                     # a directory where the file belongs
    e = _refused(DATA, envelope(DATA, [trust.r]))
    assert "could not be read" in e.summary and str(f) in e.reason


def test_saved_keyset_older_than_pinned_is_ignored(trust, home, monkeypatch):
    a, b, _c = trust.recovery
    r_old = Ed25519PrivateKey.generate()
    f = _trust_file(home)
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(keyset(1, [r_old], trust.recovery, signers=[a, b])),
                 encoding="utf-8")
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(2, [trust.r], trust.recovery))
    assert updater.verify_signature(DATA, envelope(DATA, [trust.r])) is None
    _refused(DATA, envelope(DATA, [r_old]))


def test_verify_signature_never_writes(trust, home):
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, b])
    assert updater.verify_signature(DATA, envelope(DATA, [r2], keyset=ks)) is not None
    assert not home.exists()


def test_explicit_trusted_keyset_is_used_and_validated(trust):
    other = Ed25519PrivateKey.generate()
    given = keyset(1, [other], trust.recovery)
    assert updater.verify_signature(DATA, envelope(DATA, [other]), trusted=given) is None
    _refused(DATA, envelope(DATA, [trust.r]), trusted=given)
    with pytest.raises(ValueError):
        updater.verify_signature(DATA, envelope(DATA, [other]),
                                 trusted=keyset(1, [other], trust.recovery,
                                                recovery_threshold=1))


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
def apply_cfg(monkeypatch, home):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")


def _apply(inst, data, sig):
    return updater.apply(5, installed=inst, signature=sig,
                         download_opener=_opener_for(data), runner=lambda c: 0)


def test_apply_installs_a_release_signed_build(tmp_path, trust, apply_cfg, home):
    inst = _fake_install(tmp_path)
    data = _build_zip_bytes("0.2.0")
    res = _apply(inst, data, envelope(data, [trust.r]))
    assert res["applied"] and (inst / "VERSION").read_text().strip() == "0.2.0"
    assert not _trust_file(home).exists()


def test_apply_refuses_unsigned_build_before_swap(tmp_path, trust, apply_cfg):
    inst = _fake_install(tmp_path)
    with pytest.raises(LocalmError):
        _apply(inst, _build_zip_bytes("0.2.0"), None)
    _assert_untouched(inst)


def test_apply_refuses_tampered_build_before_swap(tmp_path, trust, apply_cfg):
    inst = _fake_install(tmp_path)
    good = _build_zip_bytes("0.2.0", deps=("click",))
    tampered = _build_zip_bytes("0.2.0", deps=("click", "evil"))
    with pytest.raises(LocalmError):
        _apply(inst, tampered, envelope(good, [trust.r]))
    _assert_untouched(inst)


def test_apply_refuses_downgrade_before_swap(tmp_path, trust, apply_cfg, monkeypatch):
    monkeypatch.setattr(_version, "read_version", lambda: "0.9.0")
    inst = _fake_install(tmp_path)
    old_build = _build_zip_bytes("0.2.0")
    with pytest.raises(LocalmError):
        _apply(inst, old_build, envelope(old_build, [trust.r]))
    _assert_untouched(inst)


def test_apply_rotation_persists_and_revokes_the_old_release_key(
        tmp_path, trust, apply_cfg, home, monkeypatch):
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, b])
    inst = _fake_install(tmp_path)
    data = _build_zip_bytes("0.2.0")
    _apply(inst, data, envelope(data, [r2], keyset=ks))
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    saved = json.loads(_trust_file(home).read_text(encoding="utf-8"))
    assert saved == ks

    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    nxt = _build_zip_bytes("0.3.0")
    with pytest.raises(LocalmError):
        _apply(inst, nxt, envelope(nxt, [trust.r]))      # the replaced key
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    _apply(inst, nxt, envelope(nxt, [r2]))                # no key set needed now
    assert (inst / "VERSION").read_text().strip() == "0.3.0"


@pytest.mark.parametrize("case", ["tampered", "downgrade", "bad_zip"])
def test_refused_update_leaves_trust_unchanged(tmp_path, trust, apply_cfg, home,
                                               monkeypatch, case):
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, b])
    inst = _fake_install(tmp_path)
    data = _build_zip_bytes("0.2.0")
    sig = envelope(data, [r2], keyset=ks)
    if case == "tampered":
        data = _build_zip_bytes("0.2.0", deps=("click", "evil"))
        sig = envelope(_build_zip_bytes("0.2.0"), [r2], keyset=ks)
    elif case == "downgrade":
        monkeypatch.setattr(_version, "read_version", lambda: "0.9.0")
    else:
        data = b"not a zip"
        sig = envelope(data, [r2], keyset=ks)
    with pytest.raises(ValueError if case == "bad_zip" else LocalmError):
        _apply(inst, data, sig)
    assert not _trust_file(home).exists()
    _assert_untouched(inst)


def test_failed_keyset_save_refuses_the_update(tmp_path, trust, apply_cfg, home,
                                               monkeypatch):
    import os
    a, b, _c = trust.recovery
    r2 = Ed25519PrivateKey.generate()
    ks = keyset(2, [r2], trust.recovery, signers=[a, b])
    inst = _fake_install(tmp_path)
    data = _build_zip_bytes("0.2.0")

    def no_replace(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", no_replace)
    with pytest.raises(LocalmError) as ei:
        _apply(inst, data, envelope(data, [r2], keyset=ks))
    assert "could not save" in ei.value.summary
    _assert_untouched(inst)
    assert not _trust_file(home).exists()
    assert not [p for p in _trust_file(home).parent.iterdir() if p.name.endswith(".tmp")]


# ---------------- end-to-end: the maintainer's signer -------------------

def _load_sign_release(monkeypatch):
    import importlib.util
    from pathlib import Path
    script = Path(updater.__file__).resolve().parents[1] / "scripts" / "sign_release.py"
    if not script.is_file():
        pytest.skip(f"{script} not present (gitignored maintainer-only release "
                    "tooling) - skipping the sign_release.py round-trip")
    spec = importlib.util.spec_from_file_location("sign_release", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    checked = []
    monkeypatch.setattr(mod, "check_key_permissions", lambda key, **k: checked.append(key))
    return mod, checked


def _pem_pub(path):
    return pub_hex(_ser.load_pem_private_key(path.read_bytes(), password=None))


def test_sign_release_script_roundtrip(tmp_path, monkeypatch, home):
    """sign_release.py generates keys, signs a key set with two recovery keys and a
    build with the new release key, and the updater verifies the result."""
    mod, checked = _load_sign_release(monkeypatch)
    keys = {}
    for name in ("r1", "r2", "a", "b", "c"):
        keys[name] = tmp_path / f"{name}.pem"
        assert mod.main(["--generate-key", "--out", str(keys[name])]) == 0
    assert mod.main(["--generate-key", "--out", str(keys["a"])]) == 2   # never overwrite
    pub = {n: _pem_pub(p) for n, p in keys.items()}
    monkeypatch.setattr(updater, "_PINNED_KEYSET", {
        "seq": 1, "release": {"threshold": 1, "keys": [pub["r1"]]},
        "recovery": {"threshold": 2, "keys": [pub["a"], pub["b"], pub["c"]]}})

    build = tmp_path / "build.zip"
    build.write_bytes(_build_zip_bytes("0.2.0"))
    assert mod.main(["sign", str(build), "--key", str(keys["r1"])]) == 0
    sig = (tmp_path / "build.zip.sig").read_text(encoding="utf-8")
    assert updater.verify_signature(build.read_bytes(), sig) is None

    ks_file = tmp_path / "keyset.json"
    assert mod.main(["keyset", "--seq", "2", "--release", pub["r2"],
                     "--recovery", pub["a"], "--recovery", pub["b"], "--recovery", pub["c"],
                     "--key", str(keys["a"]), "--key", str(keys["c"]),
                     "--out", str(ks_file)]) == 0
    out = tmp_path / "rotated.sig"
    assert mod.main(["sign", str(build), "--key", str(keys["r2"]),
                     "--keyset", str(ks_file), "--out", str(out)]) == 0
    accepted = updater.verify_signature(build.read_bytes(), out.read_text(encoding="utf-8"))
    assert accepted["seq"] == 2 and accepted["release"]["keys"] == [pub["r2"]]
    assert {keys["r1"], keys["r2"], keys["a"], keys["c"]} <= set(checked)


def test_sign_release_keyset_needs_two_distinct_keys(tmp_path, monkeypatch):
    mod, _ = _load_sign_release(monkeypatch)
    a = tmp_path / "a.pem"
    assert mod.main(["--generate-key", "--out", str(a)]) == 0
    p = _pem_pub(a)
    copy = tmp_path / "a-copy.pem"
    copy.write_bytes(a.read_bytes())
    base = ["keyset", "--seq", "2", "--release", keypair()[1], "--recovery", p,
            "--recovery", keypair()[1], "--out", str(tmp_path / "k.json")]
    assert mod.main(base + ["--key", str(a)]) == 2
    assert mod.main(base + ["--key", str(a), "--key", str(a)]) == 2
    assert mod.main(base + ["--key", str(a), "--key", str(copy)]) == 2
    assert not (tmp_path / "k.json").exists()


def test_sign_release_refuses_a_malformed_keyset_file(tmp_path, monkeypatch):
    mod, _ = _load_sign_release(monkeypatch)
    r = tmp_path / "r.pem"
    assert mod.main(["--generate-key", "--out", str(r)]) == 0
    build = tmp_path / "build.zip"
    build.write_bytes(b"zip")
    bad = tmp_path / "keyset.json"
    bad.write_text(json.dumps({"seq": 2}), encoding="utf-8")
    assert mod.main(["sign", str(build), "--key", str(r), "--keyset", str(bad)]) == 2
    assert not (tmp_path / "build.zip.sig").exists()
