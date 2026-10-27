# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installing a chosen release (``localm update --list`` / ``--to``, the GUI picker):
the version list, a selected install in either direction, the warn-and-confirm
path for a build whose signature does not verify, the trust carried to an older
build, and the owner gate on the HTTP route. The plain update path keeps its
anti-rollback and mandatory signature check."""

import base64
import importlib.util
import io
import json
import zipfile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import localm
from localm import _version, updater
from localm.bugreport import LocalmError
from tests._net_config import set_config
from tests._update_signing import envelope, keyset

_PRIV = Ed25519PrivateKey.generate()
_RECOVERY = [Ed25519PrivateKey.generate() for _ in range(3)]


def _opener(payload, status=200):
    """A _proxy.request opener recording the url it was asked for."""
    def op(method, url, data, headers, timeout):
        op.urls.append(url)
        return status, json.dumps(payload).encode("utf-8")
    op.urls = []
    return op


def _zip(version, *, files=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version)
        z.writestr("pyproject.toml", '[project]\nname = "localm"\ndependencies = ["click"]\n')
        z.writestr("localm/__init__.py", f"# {version}")
        for name, text in (files or {}).items():
            z.writestr(name, text)
    return buf.getvalue()


def _download(data):
    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return opener


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A configured updater, the test key set pinned, running version 0.3.0, a
    fake install at 0.3.0 and a data dir holding a config.json."""
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text('{"kv_cache_type": "q8_0"}', encoding="utf-8")
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(1, [_PRIV], _RECOVERY))
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.0")
    inst = tmp_path / "inst"
    (inst / "localm").mkdir(parents=True)
    (inst / "localm" / "__init__.py").write_text("# 0.3.0", encoding="utf-8")
    (inst / "VERSION").write_text("0.3.0", encoding="utf-8")
    (inst / "pyproject.toml").write_text(
        '[project]\nname = "localm"\ndependencies = ["click"]\n', encoding="utf-8")
    return home, inst


def _installed(inst):
    return (inst / "VERSION").read_text(encoding="utf-8").strip()


# ------------------------------- the list --------------------------------

def test_list_versions_marks_direction_and_warnings(env, monkeypatch):
    op = _opener({"ok": True, "releases": [
        {"version": "v0.4.0", "published_at": "2026-11-01", "asset": {"id": 4}, "signed": True},
        {"version": "v0.3.0", "asset": {"id": 3}, "signed": True},
        {"version": "v0.2.0", "asset": {"id": 2}, "signed": True},
        {"version": "v0.1.9", "asset": None, "signed": False},
        {"version": "nightly", "asset": {"id": 9}, "signed": True},
        "junk", {"no_version": True},
    ]})
    out = updater.list_versions(opener=op)
    assert op.urls == ["https://w/update/releases"]
    by = {r["version"]: r for r in out["releases"]}
    assert out["current"] == "0.3.0"
    assert list(by) == ["v0.4.0", "v0.3.0", "v0.2.0", "v0.1.9", "nightly"]
    assert by["v0.4.0"]["direction"] == "newer" and by["v0.4.0"]["warnings"] == []
    assert by["v0.3.0"]["direction"] == "same"
    assert by["v0.2.0"]["direction"] == "older" and by["v0.2.0"]["installable"] is True
    assert any("downgrade" in w for w in by["v0.2.0"]["warnings"])
    assert any("does not accept" in w for w in by["v0.2.0"]["warnings"])

    assert by["v0.1.9"]["installable"] is False
    assert "no build" in by["v0.1.9"]["reason"]
    assert any("no signature" in w for w in by["v0.1.9"]["warnings"])
    assert by["nightly"]["direction"] == "unknown"


def test_list_versions_asks_for_prereleases_only_when_opted_in(env, monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w",
                                     "update_allow_prerelease": True})
    op = _opener({"ok": True, "releases": []})
    updater.list_versions(opener=op)
    assert op.urls == ["https://w/update/releases?channel=prerelease"]


def test_list_versions_on_a_proxy_without_the_route_says_so(env):
    with pytest.raises(LocalmError) as ei:
        updater.list_versions(opener=_opener({"error": "not found"}, status=404))
    assert "does not offer a choice of versions" in ei.value.summary


def test_list_versions_blocked_when_network_is_off(env, monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w", "net_mode": "off"})
    monkeypatch.setattr("localm.netpolicy.network_mode", lambda: "off")
    op = _opener({"ok": True, "releases": []})
    with pytest.raises(LocalmError):
        updater.list_versions(opener=op)
    assert op.urls == []


def test_fetch_release_found_and_not_found(env):
    op = _opener({"ok": True, "found": True, "version": "v0.2.0",
                  "asset": {"id": 2}, "signature": "S"})
    rel = updater.fetch_release("0.2.0", opener=op)
    assert op.urls == ["https://w/update/release?version=0.2.0"]
    assert rel["asset"]["id"] == 2 and rel["signature"] == "S"
    with pytest.raises(LocalmError) as ei:
        updater.fetch_release("9.9.9", opener=_opener({"ok": True, "found": False}))
    assert "no release 9.9.9" in ei.value.summary


def test_fetch_release_refuses_a_different_release_than_the_one_asked_for(env):
    op = _opener({"ok": True, "found": True, "version": "v0.3.0",
                  "asset": {"id": 3}, "signature": "S"})
    with pytest.raises(LocalmError) as ei:
        updater.fetch_release("0.3.1", opener=op)
    assert "answered v0.3.0 for 0.3.1" in ei.value.summary


def test_data_backups_in_the_same_second_never_replace_each_other(env):
    home, _inst = env
    first = updater._backup_data_for_downgrade("0.2.0")
    second = updater._backup_data_for_downgrade("0.2.0")
    assert first != second
    assert (first / "config.json").is_file() and (second / "config.json").is_file()


# ------------------------- a selected install ------------------------------

def test_selected_older_version_installs_and_backs_up_data(env):
    home, inst = env
    data = _zip("0.2.0")
    res = updater.apply(2, signature=envelope(data, [_PRIV]), target_version="v0.2.0",
                        installed=inst, download_opener=_download(data))
    assert _installed(inst) == "0.2.0"
    assert res["applied"] and res["downgrade"] is True and res["verified"] is True
    from pathlib import Path
    backup = Path(res["data_backup"])
    assert backup.parent == home / "updates"
    assert json.loads((backup / "config.json").read_text(encoding="utf-8")) == {
        "kv_cache_type": "q8_0"}


def test_selected_newer_version_installs_without_data_backup(env):
    _home, inst = env
    data = _zip("0.4.0")
    res = updater.apply(4, signature=envelope(data, [_PRIV]), target_version="0.4.0",
                        installed=inst, download_opener=_download(data))
    assert _installed(inst) == "0.4.0"
    assert res["downgrade"] is False and res["data_backup"] is None


def test_plain_update_still_refuses_a_downgrade(env):
    _home, inst = env
    data = _zip("0.2.0")
    with pytest.raises(LocalmError) as ei:
        updater.apply(2, signature=envelope(data, [_PRIV]), installed=inst,
                      download_opener=_download(data))
    assert "not newer" in ei.value.summary
    assert _installed(inst) == "0.3.0"


def test_plain_update_still_refuses_an_unverified_build(env):
    _home, inst = env
    data = _zip("0.4.0")
    with pytest.raises(LocalmError):
        updater.apply(4, signature=envelope(b"other bytes", [_PRIV]), installed=inst,
                      download_opener=_download(data))
    assert _installed(inst) == "0.3.0"


def test_selected_version_mismatch_is_refused(env):
    _home, inst = env
    data = _zip("0.1.9")
    with pytest.raises(LocalmError) as ei:
        updater.apply(2, signature=envelope(data, [_PRIV]), target_version="0.2.0",
                      installed=inst, download_opener=_download(data))
    assert "it is 0.1.9" in ei.value.summary
    assert _installed(inst) == "0.3.0"


def test_selected_mismatch_is_refused_even_when_unverified_is_allowed(env):
    _home, inst = env
    data = _zip("0.1.9")
    import hashlib
    with pytest.raises(LocalmError):
        updater.apply(2, signature=None, target_version="0.2.0", installed=inst,
                      allow_unverified=True, expect_sha256=hashlib.sha256(data).hexdigest(),
                      download_opener=_download(data))
    assert _installed(inst) == "0.3.0"


def test_rollback_after_a_selected_downgrade_returns_to_the_previous_version(env):
    _home, inst = env
    data = _zip("0.2.0")
    updater.apply(2, signature=envelope(data, [_PRIV]), target_version="0.2.0",
                  installed=inst, download_opener=_download(data))
    assert _installed(inst) == "0.2.0"
    updater.rollback_last(installed=inst)
    assert _installed(inst) == "0.3.0"
    assert (inst / "localm" / "__init__.py").read_text(encoding="utf-8") == "# 0.3.0"


# ---------------------- a build that does not verify ----------------------

def test_tampered_build_warns_with_reason_then_installs_on_consent(env):
    home, inst = env
    data = _zip("0.2.0")
    sig = envelope(b"the bytes that were really signed", [_PRIV])
    with pytest.raises(updater.UnverifiedBuild) as ei:
        updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                      download_opener=_download(data))
    assert _installed(inst) == "0.3.0", "nothing is installed before the user agrees"
    assert "altered" in ei.value.reason
    import hashlib
    assert ei.value.sha256 == hashlib.sha256(data).hexdigest()
    res = updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                        allow_unverified=True, expect_sha256=ei.value.sha256,
                        download_opener=_download(data))
    assert _installed(inst) == "0.2.0"
    assert res["verified"] is False and "altered" in res["verify_reason"]
    assert not (home / "updates" / "trust.json").exists()


def test_consent_is_bound_to_the_confirmed_build(env):
    _home, inst = env
    data = _zip("0.2.0")
    with pytest.raises(LocalmError) as ei:
        updater.apply(2, signature=None, target_version="0.2.0", installed=inst,
                      allow_unverified=True, expect_sha256="0" * 64,
                      download_opener=_download(data))
    assert "differs from the one you confirmed" in ei.value.summary
    assert _installed(inst) == "0.3.0"


def test_allow_unverified_needs_the_sha256(env):
    _home, inst = env
    data = _zip("0.2.0")
    with pytest.raises(LocalmError):
        updater.apply(2, signature=None, target_version="0.2.0", installed=inst,
                      allow_unverified=True, download_opener=_download(data))
    assert _installed(inst) == "0.3.0"


def test_unsigned_build_reason(env):
    _home, inst = env
    data = _zip("0.2.0")
    with pytest.raises(updater.UnverifiedBuild) as ei:
        updater.apply(2, signature=None, target_version="0.2.0", installed=inst,
                      download_opener=_download(data))
    assert "no signature" in ei.value.reason


def test_a_legacy_bare_signature_is_reported_and_never_trusted(env):
    _home, inst = env
    data = _zip("0.2.0")
    bare = base64.b64encode(Ed25519PrivateKey.generate().sign(data)).decode("ascii")
    with pytest.raises(updater.UnverifiedBuild) as ei:
        updater.apply(2, signature=bare, target_version="0.2.0", installed=inst,
                      download_opener=_download(data))
    assert "0.2.0 or earlier" in ei.value.reason
    with pytest.raises(LocalmError):
        updater.verify_signature(data, bare)
    assert _installed(inst) == "0.3.0"


def test_selected_install_never_moves_the_key_set_back(env):
    home, inst = env
    newer = keyset(2, [_PRIV], _RECOVERY, signers=_RECOVERY[:2])
    updater._save_keyset(newer)
    data = _zip("0.2.0")
    old = keyset(1, [_PRIV], _RECOVERY, signers=_RECOVERY[:2])
    sig = envelope(data, [_PRIV], keyset=old)
    with pytest.raises(updater.UnverifiedBuild) as ei:
        updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                      download_opener=_download(data))
    assert "older than the trusted key set 2" in ei.value.reason
    updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                  allow_unverified=True, expect_sha256=ei.value.sha256,
                  download_opener=_download(data))
    saved = json.loads((home / "updates" / "trust.json").read_text(encoding="utf-8"))
    assert saved["seq"] == 2


def test_unverified_build_never_installs_its_key_set(env):
    home, inst = env
    data = _zip("0.2.0")
    offered = keyset(2, [_PRIV], _RECOVERY, signers=_RECOVERY[:2])
    sig = envelope(b"not these bytes", [_PRIV], keyset=offered)
    with pytest.raises(updater.UnverifiedBuild) as ei:
        updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                      download_opener=_download(data))
    updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                  allow_unverified=True, expect_sha256=ei.value.sha256,
                  download_opener=_download(data))
    assert not (home / "updates" / "trust.json").exists()


# --------------------- an older build can update again ---------------------

_OLD_UPDATER = '''
def verify_signature(data, signature_b64):
    raise RuntimeError("the old verifier trusts only a retired key")


def check_build(data, signature):
    return verify_signature(data, signature)
'''


def _load_installed_old_updater(inst, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "localm._update_trust_bridge", inst / "localm" / "_update_trust_bridge.py")
    bridge = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(__import__("sys").modules, "localm._update_trust_bridge", bridge)
    monkeypatch.setattr(localm, "_update_trust_bridge", bridge, raising=False)
    spec.loader.exec_module(bridge)
    spec = importlib.util.spec_from_file_location(
        "old_updater_under_test", inst / "localm" / "updater.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    return old


def test_older_build_gets_a_trust_bridge_and_verifies_newer_releases(env, monkeypatch):
    home, inst = env
    data = _zip("0.2.0", files={"localm/updater.py": _OLD_UPDATER})
    sig = base64.b64encode(Ed25519PrivateKey.generate().sign(data)).decode("ascii")
    with pytest.raises(updater.UnverifiedBuild) as ei:
        updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                      download_opener=_download(data))
    updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                  allow_unverified=True, expect_sha256=ei.value.sha256,
                  download_opener=_download(data))
    assert _installed(inst) == "0.2.0"
    assert (inst / "localm" / "_update_trust_bridge.py").is_file()

    old = _load_installed_old_updater(inst, monkeypatch)
    newer = _zip("0.4.0")
    assert old.check_build(newer, envelope(newer, [_PRIV])) is None
    with pytest.raises(LocalmError):
        old.check_build(newer, envelope(b"tampered", [_PRIV]))
    rotated = keyset(2, [_PRIV], _RECOVERY, signers=_RECOVERY[:2])
    old.check_build(newer, envelope(newer, [_PRIV], keyset=rotated))
    saved = json.loads((home / "updates" / "trust.json").read_text(encoding="utf-8"))
    assert saved["seq"] == 2, "a key rotation the older build accepts is kept"


_OLD_APPLY_UPDATE = '''
def post_swap_command(klass, backend=None):
    if klass == "deps":
        return ["uv", "pip", "install", "-p", ".venv", "-e", ".[coder,voice,monitor]"]
    return None
'''


def test_older_build_runs_its_dependency_step_with_the_found_uv(env, monkeypatch):
    import subprocess
    import sys
    _home, inst = env
    contained = inst / ".uv" / "uv.exe"
    contained.parent.mkdir()
    contained.write_bytes(b"")
    found = []
    monkeypatch.setattr(updater, "_uv_for_deps", lambda t: found.append(t) or str(contained))
    data = _zip("0.2.0", files={"localm/updater.py": _OLD_UPDATER,
                                "localm/_apply_update.py": _OLD_APPLY_UPDATE})
    sig = envelope(data, [_PRIV])
    updater.apply(2, signature=sig, target_version="0.2.0", installed=inst,
                  download_opener=_download(data))
    assert found == [inst]
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import localm.updater; "
             "from localm import _apply_update as a; "
             "print(a.post_swap_command('deps')[0]); print(a.post_swap_command('reboot'))")
    out = subprocess.run([sys.executable, "-I", "-c", probe, str(inst)],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    from pathlib import Path
    assert Path(out[0]).resolve() == contained.resolve(), out
    assert out[1] == "None"


def test_a_build_that_needs_no_uv_bridge_gets_none(env, monkeypatch):
    _home, inst = env
    monkeypatch.setattr(updater, "_uv_for_deps",
                        lambda t: pytest.fail("no uv lookup for a build that needs none"))
    current = "def post_swap_command(klass, backend=None, uv=None):\n    return [uv]\n"
    data = _zip("0.2.0", files={"localm/_apply_update.py": current,
                                "localm/updater.py": _OLD_UPDATER})
    updater.apply(2, signature=envelope(data, [_PRIV]), target_version="0.2.0",
                  installed=inst, download_opener=_download(data))
    assert "_localm_bridged_post_swap_command" not in (
        inst / "localm" / "updater.py").read_text(encoding="utf-8")


def test_a_build_that_already_verifies_envelopes_gets_no_bridge(env):
    _home, inst = env
    data = _zip("0.2.0", files={
        "localm/updater.py": '_SIG_FORMAT = "localm-update-sig/2"\n'})
    updater.apply(2, signature=envelope(data, [_PRIV]), target_version="0.2.0",
                  installed=inst, download_opener=_download(data))
    assert not (inst / "localm" / "_update_trust_bridge.py").exists()


def test_a_downgrade_carries_a_newer_trusted_key_set_into_the_data_dir(env, monkeypatch):
    home, inst = env
    monkeypatch.setattr(updater, "_PINNED_KEYSET", keyset(2, [_PRIV], _RECOVERY))
    data = _zip("0.2.0")
    updater.apply(2, signature=envelope(data, [_PRIV]), target_version="0.2.0",
                  installed=inst, download_opener=_download(data))
    saved = json.loads((home / "updates" / "trust.json").read_text(encoding="utf-8"))
    assert saved["seq"] == 2


# ---------------------------- native runtime -------------------------------

def test_a_different_llama_pin_reprovisions_the_runtime(env, monkeypatch):
    _home, inst = env
    (inst / "localm" / "setup_llama").mkdir()
    (inst / "localm" / "setup_llama" / "pins.py").write_text(
        '_PINNED_TAG = "b11757"\n', encoding="utf-8")
    data = _zip("0.2.0", files={"localm/setup_llama/pins.py": '_PINNED_TAG = "b11541"\n'})
    ran = []
    res = updater.apply(2, signature=envelope(data, [_PRIV]), target_version="0.2.0",
                        installed=inst, download_opener=_download(data),
                        runner=lambda c: ran.append(c) or 0)
    assert res["klass"] == "runtime"
    assert ran and "setup-llama" in ran[0]
    assert "--keep-choice" not in ran[0], "that build's setup-llama has no such flag"


def test_the_same_llama_pin_keeps_the_runtime(env):
    _home, inst = env
    data = _zip("0.2.0")
    res = updater.apply(2, signature=envelope(data, [_PRIV]), target_version="0.2.0",
                        installed=inst, download_opener=_download(data),
                        runner=lambda c: pytest.fail(f"unexpected post step {c}"))
    assert res["klass"] == "reboot"


# ------------------------------ HTTP routes --------------------------------

OWNER_KEY = "owner-admin-key-select-version-abc123"


@pytest.fixture
def routes(tmp_path, monkeypatch):
    """Protected mode with an owner key and a scoped config:read/write key; the
    updater's proxy calls and apply are stubbed and recorded."""
    from fastapi.testclient import TestClient

    import localm.config as cfg
    import localm.inference.http_server as _hs
    from localm import auth
    from localm.inference.http_server import create_app
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    restarts, applied = [], []
    monkeypatch.setattr(_hs, "_request_restart", lambda *a, **k: restarts.append(k))
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "fetch_release", lambda v: {
        "version": "v" + v.lstrip("v"), "asset": {"id": 7}, "signature": "SIG"})

    def fake_apply(asset_id, **kw):
        applied.append((asset_id, kw))
        if kw.get("signature") == "SIG" and not kw.get("allow_unverified"):
            raise updater.UnverifiedBuild(kw["target_version"], "it is unsigned", "a" * 64)
        return {"applied": True, "version": kw["target_version"].lstrip("v"),
                "klass": "reboot", "verified": False}
    monkeypatch.setattr(updater, "apply", fake_apply)
    monkeypatch.setattr(updater, "list_versions", lambda: {
        "current": "0.3.0", "releases": [{"version": "v0.2.0", "direction": "older"}]})
    scoped = auth.create_key(
        "dev", ["config:read", "config:write"], allow_privileged=True)["key"]
    with TestClient(create_app(None)) as c:
        yield c, scoped, applied, restarts


def _bearer(key):
    return {"Authorization": f"Bearer {key}"}


def test_route_lists_releases(routes):
    c, scoped, _applied, _restarts = routes
    r = c.get("/api/update/releases", headers=_bearer(scoped))
    assert r.status_code == 200, r.text
    assert r.json()["releases"][0]["version"] == "v0.2.0"
    assert r.json()["available"] is True


def test_route_selected_install_is_owner_only(routes):
    c, scoped, applied, restarts = routes
    r = c.post("/api/update/apply", headers=_bearer(scoped), json={"version": "0.2.0"})
    assert applied == [] and restarts == []
    assert r.status_code == 403, r.text
    assert "owner" in r.text.lower()


def test_route_selected_install_asks_then_installs_the_confirmed_build(routes):
    c, _scoped, applied, restarts = routes
    r = c.post("/api/update/apply", headers=_bearer(OWNER_KEY), json={"version": "0.2.0"})
    assert restarts == []
    body = r.json()
    assert body["applied"] is False and body["needs_confirmation"] == "unverified"
    assert body["reason"] == "it is unsigned" and body["sha256"] == "a" * 64
    assert applied[0][0] == 7 and applied[0][1]["target_version"] == "v0.2.0"
    r = c.post("/api/update/apply", headers=_bearer(OWNER_KEY), json={
        "version": "0.2.0", "allow_unverified": True, "sha256": body["sha256"]})
    assert r.json()["applied"] is True and r.json()["restarting"] is True
    assert applied[1][1]["expect_sha256"] == "a" * 64
    assert applied[1][1]["allow_unverified"] is True
    assert len(restarts) == 1


def test_route_rejects_allow_unverified_without_sha256_and_bad_bodies(routes):
    c, _scoped, applied, _restarts = routes
    for body in ({"version": "0.2.0", "allow_unverified": True},
                 {"version": "0.2.0", "sha256": "xyz"},
                 {"version": ""}, {"version": 5}):
        r = c.post("/api/update/apply", headers=_bearer(OWNER_KEY), json=body)
        assert r.status_code == 400, (body, r.text)
    for raw in (b"{not json", b"[" * 100000):
        r = c.post("/api/update/apply", headers={**_bearer(OWNER_KEY),
                   "Content-Type": "application/json"}, content=raw)
        assert r.status_code == 400, raw[:10]
    assert applied == []
