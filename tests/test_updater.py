# SPDX-License-Identifier: AGPL-3.0-or-later
"""The self-updater: configured endpoint resolution, the version check, update
classification (reboot/deps/runtime/setup), and the build download. Apply is NOT
exercised here (it has its own detached-helper tests)."""

import base64
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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


def test_classify_setup_when_requires_python_removed(tmp_path):
    # Installed pins requires-python; the staged build drops it -> still a setup change.
    inst, stg = tmp_path / "i", tmp_path / "s"
    inst.mkdir(); stg.mkdir()
    _write_pyproject(inst, ["click"], requires_python="==3.12.*")
    (stg / "pyproject.toml").write_text(
        '[project]\nname = "localm"\ndependencies = ["click"]\n', encoding="utf-8")
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


def test_download_refuses_non_https_endpoint(tmp_path, monkeypatch):
    """CHK-UPDATER-INTEGRITY (transport): a code-update download over the real
    urllib path must refuse a non-HTTPS endpoint (no cleartext code delivery)."""
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "http://insecure.example"})   # http, not https
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.download(7, tmp_path / "build.zip")           # opener=None -> urllib path


# ------------------------- apply / rollback -----------------------------
#
# apply() verifies an Ed25519 signature over the downloaded build against a pinned
# public key BEFORE extracting/swapping, and refuses a non-newer build (anti-
# rollback). These tests pin a THROWAWAY keypair and sign the fixture build, so they
# exercise the real gate rather than disabling it (see test_updater_signature.py for
# the negative cases: tampered / unsigned / no-key / downgrade).

_TEST_PRIV = Ed25519PrivateKey.generate()
_TEST_PUB_HEX = _TEST_PRIV.public_key().public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw).hex()


def _pyproject_str(deps):
    return ('[project]\nname = "localm"\ndependencies = ['
            + ", ".join(f'"{d}"' for d in deps) + ']\n')


def _signed_build(version, deps):
    """A deterministic build zip (bytes) + its base64 Ed25519 signature under the
    test key. Returns ``(download_opener, signature_b64)``; the opener writes the
    exact bytes that were signed, so the gate verifies them."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version)
        z.writestr("pyproject.toml", _pyproject_str(deps))
        z.writestr("localm/__init__.py", f"# {version}")
    data = buf.getvalue()
    sig = base64.b64encode(_TEST_PRIV.sign(data)).decode("ascii")

    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return opener, sig


@pytest.fixture
def sig_env(monkeypatch):
    """Pin the test signing key and an OLDER running version so apply()'s signature
    gate and anti-rollback accept the 0.2.0 fixture builds."""
    monkeypatch.setattr(updater, "_UPDATE_PUBKEYS", (_TEST_PUB_HEX,))
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")


def _fake_install(tmp_path, deps=("click",)):
    inst = tmp_path / "inst"
    (inst / "localm").mkdir(parents=True)
    (inst / "localm" / "__init__.py").write_text("# 0.1.0", encoding="utf-8")
    (inst / "VERSION").write_text("0.1.0", encoding="utf-8")
    (inst / "pyproject.toml").write_text(_pyproject_str(deps), encoding="utf-8")
    return inst


def test_apply_reboot_swaps_and_reports(tmp_path, monkeypatch, sig_env):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    op, sig = _signed_build("0.2.0", ["click"])
    res = updater.apply(5, installed=inst, signature=sig, download_opener=op)
    assert res["applied"] is True and res["version"] == "0.2.0"
    assert res["klass"] == "reboot"  # deps unchanged
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    assert (inst / "localm" / "__init__.py").read_text() == "# 0.2.0"


def test_apply_deps_runs_post_command(tmp_path, monkeypatch, sig_env):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    ran = []
    op, sig = _signed_build("0.2.0", ["click", "httpx"])
    res = updater.apply(5, installed=inst, signature=sig, download_opener=op,
                        runner=lambda c: ran.append(c) or 0)
    assert res["klass"] == "deps"
    assert ran and ran[0][:3] == ["uv", "pip", "install"]


def test_apply_rolls_back_on_post_command_failure(tmp_path, monkeypatch, sig_env):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    op, sig = _signed_build("0.2.0", ["click", "httpx"])
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.apply(5, installed=inst, signature=sig, download_opener=op,
                      runner=lambda c: 1)  # post step fails -> rollback
    # Rolled back to the pre-apply state.
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    assert (inst / "localm" / "__init__.py").read_text() == "# 0.1.0"


def test_rollback_last_restores(tmp_path, monkeypatch, sig_env):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    op, sig = _signed_build("0.2.0", ["click"])
    updater.apply(5, installed=inst, signature=sig, download_opener=op)
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    updater.rollback_last(installed=inst)
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    assert (inst / "localm" / "__init__.py").read_text() == "# 0.1.0"


def test_apply_surfaces_rollback_failure(tmp_path, monkeypatch, sig_env):
    """If the post-update step fails AND rollback also fails, apply() must say so
    (manual recovery needed) - never report a clean rollback over a broken install."""
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    from localm import _apply_update as au
    from localm.bugreport import LocalmError

    def rb_fail(*a, **k):
        raise OSError("rollback boom")

    monkeypatch.setattr(au, "rollback", rb_fail)
    op, sig = _signed_build("0.2.0", ["click", "httpx"])   # deps -> post cmd
    with pytest.raises(LocalmError, match="rollback failed"):
        updater.apply(5, installed=inst, signature=sig, download_opener=op,
                      runner=lambda c: 1)  # post step fails -> rollback attempted -> also fails


def test_rollback_last_without_backup_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home2")
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.rollback_last(installed=tmp_path / "inst")


def _stage_rollback(tmp_path, monkeypatch, *, manifest):
    """Build a fake updates dir (backup + optional applied-names manifest) and an install
    where an update replaced `existing.txt` and ADDED `brand_new/`, so a rollback must
    restore the old file and (with a manifest) remove the added entry."""
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    updir = home / "updates"
    (updir / "backup").mkdir(parents=True)
    install = tmp_path / "install"
    install.mkdir()
    # pre-existing entry the update replaced: backup holds the OLD (pre-apply) content
    (install / "existing.txt").write_text("NEW-from-update", encoding="utf-8")
    (updir / "backup" / "existing.txt").write_text("OLD-preapply", encoding="utf-8")
    # brand-new top-level entry the update ADDED (nothing was backed up for it)
    (install / "brand_new").mkdir()
    (install / "brand_new" / "f.txt").write_text("added by update", encoding="utf-8")
    if manifest is not None:
        (updir / "applied_names.json").write_text(json.dumps(manifest), encoding="utf-8")
    return install


def test_rollback_last_removes_newly_added_entry_via_manifest(tmp_path, monkeypatch):
    # The manifest records the FULL swap set, so rollback removes the brand-new entry too.
    install = _stage_rollback(tmp_path, monkeypatch,
                              manifest=["existing.txt", "brand_new"])
    res = updater.rollback_last(installed=install)
    assert res["rolled_back"] is True
    assert (install / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"  # restored
    assert not (install / "brand_new").exists()   # added entry removed -> pre-apply state


def test_rollback_last_falls_back_to_backup_listing_without_manifest(tmp_path, monkeypatch):
    # No manifest (an older backup): backed-up entries still restore correctly; the
    # brand-new entry is not known to the fallback and survives (documented limitation).
    install = _stage_rollback(tmp_path, monkeypatch, manifest=None)
    updater.rollback_last(installed=install)
    assert (install / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"  # restored
    assert (install / "brand_new").exists()   # fallback cannot remove an unrecorded new entry
