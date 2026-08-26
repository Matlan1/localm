# SPDX-License-Identifier: AGPL-3.0-or-later
"""The self-updater: configured endpoint resolution, the version check, update
classification (reboot/deps/runtime/setup), and the build download. Apply is NOT
exercised here (it has its own detached-helper tests)."""

import base64
import json
import subprocess
import sys

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


def test_check_comparable_true_for_a_real_tie(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    res = updater.check(opener=_opener({"version": "v0.2.0"}))
    assert res["newer"] is False and res["comparable"] is True


def test_check_comparable_false_for_an_unrecognized_tag(monkeypatch):
    """A tag the comparator cannot order (e.g. a "stable"/"nightly" release
    name) is reported as comparable=False rather than folded into "not
    newer"."""
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.4")
    res = updater.check(opener=_opener({"version": "nightly"}))
    assert res["newer"] is False and res["comparable"] is False


def test_check_unconfigured_raises(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.check()


# ------------------------- network policy gate ----------------------------

def _counting_opener(payload):
    """Like _opener, but records how many times it was called, so a blocked
    check can be shown never to have invoked the transport at all."""
    calls = {"n": 0}

    def op(method, url, data, headers, timeout):
        calls["n"] += 1
        op.url = url
        return 200, json.dumps(payload).encode("utf-8")

    op.calls = calls
    return op


def test_check_blocked_when_net_mode_off_never_calls_opener(monkeypatch):
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "net_mode": "off"})
    from localm.bugreport import LocalmError
    op = _counting_opener({"ok": True, "version": "v9.9.9"})
    with pytest.raises(LocalmError) as ei:
        updater.check(opener=op)
    assert op.calls["n"] == 0, "a blocked check must not attempt any connection"
    msg = f"{ei.value.summary} {ei.value.reason}".lower()
    assert "network access" in msg and "off" in msg
    assert "settings" in msg or "net_mode" in msg


def test_check_net_mode_off_but_exempted_still_calls_opener(monkeypatch):
    """The admin-only toggle lets THIS channel through even with net_mode=off -
    the opener IS invoked, and the result is a normal successful check (not the
    blocked path)."""
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "net_mode": "off",
        "update_ignore_net_policy": True})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _counting_opener({"ok": True, "version": "v0.2.0"})
    res = updater.check(opener=op)
    assert op.calls["n"] == 1
    assert res["newer"] is True
    assert op.url.endswith("/update")


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_check_ask_and_allow_modes_are_unaffected(monkeypatch, mode):
    """The gate fires only on the literal 'off' kill switch, so it is a no-op
    for the default 'ask' mode and for 'allow'."""
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "net_mode": mode})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _counting_opener({"ok": True, "version": "v0.2.0"})
    res = updater.check(opener=op)
    assert op.calls["n"] == 1
    assert res["newer"] is True


def test_check_env_var_off_blocks_even_with_config_ask(monkeypatch):
    """LOCALM_NET_MODE overrides config, matching network_mode()'s own
    precedence - a blocked check via the env var must also never call out."""
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "net_mode": "ask"})
    from localm.bugreport import LocalmError
    op = _counting_opener({"ok": True, "version": "v0.2.0"})
    with pytest.raises(LocalmError):
        updater.check(opener=op)
    assert op.calls["n"] == 0


def test_net_policy_allows_update_check_fails_safe_on_unreadable_config(monkeypatch):
    """An unreadable config resolves to BLOCKED, never exempted."""
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)

    def boom():
        raise OSError("config unreadable")

    monkeypatch.setattr("localm.config.load_config", boom)
    assert updater._net_policy_allows_update_check() is False


# ------------------------ prerelease channel -----------------------------

def test_check_stable_by_default_no_channel_param(monkeypatch):
    """Opt-in only: with the setting absent OR explicitly False, the request
    carries no query string at all, not even an empty one."""
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _opener({"ok": True, "version": "v0.2.0"})
    updater.check(opener=op)
    assert op.url.endswith("/update")
    assert "channel" not in op.url


def test_check_prerelease_opt_in_adds_channel_param(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "update_allow_prerelease": True})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _opener({"ok": True, "version": "v0.2.0"})
    updater.check(opener=op)
    assert op.url.endswith("/update?channel=prerelease")


def test_prerelease_channel_enabled_reads_the_setting(monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"update_allow_prerelease": True})
    assert updater._prerelease_channel_enabled() is True
    monkeypatch.setattr("localm.config.load_config", lambda: {"update_allow_prerelease": False})
    assert updater._prerelease_channel_enabled() is False
    monkeypatch.setattr("localm.config.load_config", lambda: {})
    assert updater._prerelease_channel_enabled() is False, "absent -> stable-only, not offered"


def test_prerelease_channel_enabled_fails_safe_to_stable(monkeypatch):
    """An unreadable config reads as 'prereleases off', never on."""
    def boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("localm.config.load_config", boom)
    assert updater._prerelease_channel_enabled() is False


def test_opting_out_after_an_rc_does_not_strand_or_downgrade(monkeypatch):
    """A client on 0.1.4-rc2 that opts back out, while the stable channel's
    latest is still older (0.1.3), stays on the rc: not newer, no downgrade
    offered, no crash."""
    monkeypatch.setattr("localm.config.load_config", lambda: {
        "bugreport_upload_url": "https://w", "update_allow_prerelease": False})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.4-rc2")
    res = updater.check(opener=_opener({"ok": True, "version": "0.1.3"}))
    assert res["newer"] is False

    # Once the matching final release ships, moving off the rc IS offered - a
    # forward upgrade, not a downgrade, via the final-outranks-its-own-
    # prerelease tie-break.
    res2 = updater.check(opener=_opener({"ok": True, "version": "0.1.4"}))
    assert res2["newer"] is True


# --------------- anti-rollback unaffected by the comparable() signal -------

def test_refuse_downgrade_unaffected_by_the_new_comparable_signal(monkeypatch):
    """_version.comparable() is a signal for CLI/API messaging only;
    _refuse_downgrade calls is_newer() directly. An unparseable *new_version* -
    attacker-controlled input to apply(), via a compromised or MITM'd proxy
    response - stays REFUSED."""
    from localm.bugreport import LocalmError
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.5")
    with pytest.raises(LocalmError):
        updater._refuse_downgrade("nightly")   # unparseable candidate: still refused
    with pytest.raises(LocalmError):
        updater._refuse_downgrade("0.1.4")     # plain older: still refused
    with pytest.raises(LocalmError):
        updater._refuse_downgrade("0.1.5")     # exact tie: still refused
    updater._refuse_downgrade("0.1.6")         # plain newer: still allowed, unchanged


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
    """A code-update download over the real urllib path refuses a non-HTTPS
    endpoint."""
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
# exercise the real gate rather than disabling it.

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


def test_apply_removes_stale_manifest_when_write_fails(tmp_path, monkeypatch, sig_env, caplog):
    """A failed applied_names.json write must NOT leave the PREVIOUS update's
    manifest in place: the updates dir persists across updates, and a stale
    manifest would make a later rollback remove or restore the WRONG top-level
    set. apply() unlinks it before the swap and WARNS on the failed write."""
    import logging
    from pathlib import Path
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    inst = _fake_install(tmp_path, deps=("click",))

    # Update #1 records a manifest for real (drives the actual swap, no mocks).
    op1, sig1 = _signed_build("0.2.0", ["click"])
    updater.apply(5, installed=inst, signature=sig1, download_opener=op1)
    manifest = home / "updates" / "applied_names.json"
    assert manifest.is_file()

    # Update #2: only the manifest write fails (the swap itself still runs for real).
    real_write = Path.write_text

    def flaky_write(self, *a, **k):
        if self.name == "applied_names.json":
            raise OSError("disk full")
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", flaky_write)
    op2, sig2 = _signed_build("0.3.0", ["click"])
    with caplog.at_level(logging.WARNING, logger="localm"):
        res = updater.apply(5, installed=inst, signature=sig2, download_opener=op2)

    assert res["applied"] is True and res["version"] == "0.3.0"
    assert not manifest.exists()   # stale #1 manifest NOT left behind (unlinked before swap)
    assert "could not record the update manifest" in caplog.text


def test_rollback_last_warns_on_corrupt_manifest_and_falls_back(tmp_path, monkeypatch, caplog):
    """A corrupt applied_names.json is not collapsed to 'absent': rollback_last
    WARNS that brand-new entries will not be removed, then falls back to the
    backup dir, still restoring the pre-existing names."""
    import logging
    install = _stage_rollback(tmp_path, monkeypatch, manifest=None)
    updir = (tmp_path / "home") / "updates"
    (updir / "applied_names.json").write_text("{ this is not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="localm"):
        updater.rollback_last(installed=install)

    assert "unreadable" in caplog.text
    assert (install / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"  # restored
    assert (install / "brand_new").exists()   # fallback cannot remove an unrecorded new entry


def test_apply_early_abort_preserves_prior_manifest(tmp_path, monkeypatch, sig_env):
    """The manifest unlink sits AFTER download/verify/extract, so an EARLY abort
    (bad signature, downgrade, corrupt zip) leaves the PREVIOUS update's
    still-valid applied_names.json intact."""
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    inst = _fake_install(tmp_path, deps=("click",))

    # A prior successful update recorded a manifest.
    op1, sig1 = _signed_build("0.2.0", ["click"])
    updater.apply(5, installed=inst, signature=sig1, download_opener=op1)
    manifest = home / "updates" / "applied_names.json"
    before = manifest.read_text(encoding="utf-8")
    assert before   # non-empty

    # A later update aborts EARLY at the signature gate (before extract/swap/unlink).
    from localm.bugreport import LocalmError
    op2, _sig2 = _signed_build("0.3.0", ["click"])
    with pytest.raises(LocalmError):
        updater.apply(5, installed=inst, signature=None, download_opener=op2)  # unsigned -> refused

    # The prior manifest is untouched: the unlink never ran (it is downstream of the gate).
    assert manifest.read_text(encoding="utf-8") == before


def test_apply_warns_when_prior_manifest_unlink_fails(tmp_path, monkeypatch, sig_env, caplog):
    """If the previous applied_names.json cannot be unlinked with a
    non-FileNotFound OSError (e.g. a lock), apply() WARNS rather than crashing,
    and the subsequent write still overwrites it."""
    import logging
    from pathlib import Path
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    inst = _fake_install(tmp_path, deps=("click",))
    op1, sig1 = _signed_build("0.2.0", ["click"])
    updater.apply(5, installed=inst, signature=sig1, download_opener=op1)
    manifest = home / "updates" / "applied_names.json"
    assert manifest.is_file()

    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **k):
        if self.name == "applied_names.json":
            raise OSError("locked by AV")   # a non-FileNotFound OSError
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    op2, sig2 = _signed_build("0.3.0", ["click"])
    with caplog.at_level(logging.WARNING, logger="localm"):
        res = updater.apply(5, installed=inst, signature=sig2, download_opener=op2)

    assert res["applied"] is True and res["version"] == "0.3.0"
    assert "could not remove the previous update manifest" in caplog.text
    assert manifest.exists()   # the write below the failed unlink still overwrote it


# --------------- apply() single-flight / concurrency ----------------------
#
# apply() takes a cross-process mkdir lock (_apply_lock) and uses a unique
# per-run scratch directory (_new_run_dir) whose backup is promoted to the
# stable path only after that run's own swap has succeeded (_promote_backup).

def _setup_apply_env(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.load_config", lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    return home, _fake_install(tmp_path, deps=("click",))


def test_apply_second_concurrent_call_refused_without_download(tmp_path, monkeypatch, sig_env):
    """A second apply() while the lock is held is refused BEFORE it downloads
    anything. Driven deterministically via re-entrancy - the outer call's own
    opener attempts a second apply() - and asserted as a call-count of zero on
    the blocked call's opener."""
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    op, sig = _signed_build("0.2.0", ["click"])
    from localm.bugreport import LocalmError

    inner_calls = {"n": 0}
    inner_error = {}

    def outer_opener(url, headers, timeout, dest):
        def inner_opener(*a, **k):
            inner_calls["n"] += 1
        try:
            updater.apply(5, installed=inst, signature=sig, download_opener=inner_opener)
        except LocalmError as e:
            inner_error["e"] = e
        op(url, headers, timeout, dest)   # let the outer call proceed normally

    updater.apply(5, installed=inst, signature=sig, download_opener=outer_opener)

    assert inner_calls["n"] == 0, "the blocked second call must never attempt a download"
    assert "already being applied" in str(inner_error.get("e", ""))


def test_apply_concurrent_threads_exactly_one_proceeds_backup_has_old_content(
        tmp_path, monkeypatch, sig_env):
    """Two REAL concurrent threads meet at the lock via a barrier, and the
    CONTENT of the backup must hold the PRE-update build, never a NEW one, so
    the losing call's backup step never ran against an already-swapped
    install."""
    import threading
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    op, sig = _signed_build("0.2.0", ["click"])
    from localm.bugreport import LocalmError

    barrier = threading.Barrier(2)
    results = {}

    def worker(key):
        barrier.wait(timeout=5)
        try:
            results[key] = ("ok", updater.apply(5, installed=inst, signature=sig,
                                                 download_opener=op))
        except Exception as e:
            results[key] = ("err", e)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    outcomes = [results.get("a", ("missing",))[0], results.get("b", ("missing",))[0]]
    assert outcomes.count("ok") == 1, f"exactly one apply should have proceeded: {results}"
    assert outcomes.count("err") == 1, f"the other must be refused, not silently dropped: {results}"
    _, err = results["a"] if results["a"][0] == "err" else results["b"]
    assert isinstance(err, LocalmError)
    assert "already being applied" in str(err)

    backup_init = home / "updates" / "backup" / "localm" / "__init__.py"
    assert backup_init.read_text(encoding="utf-8") == "# 0.1.0", \
        "the backup must hold the PRE-update build, not either NEW one"
    assert (inst / "localm" / "__init__.py").read_text(encoding="utf-8") == "# 0.2.0"


def test_apply_fresh_lock_is_not_reclaimed(tmp_path, monkeypatch, sig_env):
    """No pid recorded (an older-format lock) -> the age-only FALLBACK path;
    fresh age -> not stale."""
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    from localm.bugreport import LocalmError
    (home / "updates" / "apply.lock").mkdir(parents=True)   # fresh - well within the stale window

    called = {"n": 0}
    with pytest.raises(LocalmError, match="already being applied"):
        updater.apply(5, installed=inst, signature="x",
                      download_opener=lambda *a: called.__setitem__("n", called["n"] + 1))
    assert called["n"] == 0


def test_apply_stale_lock_is_reclaimed(tmp_path, monkeypatch, sig_env):
    """No pid recorded (an older-format lock, or a crash between mkdir and the
    pid write) -> falls back to age. A lock older than the threshold must not
    strand every future update forever."""
    import os
    import time as _time
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    monkeypatch.setattr(updater, "_APPLY_LOCK_STALE_S", 0.01)
    op, sig = _signed_build("0.2.0", ["click"])

    lock_dir = home / "updates" / "apply.lock"
    lock_dir.mkdir(parents=True)
    old = _time.time() - 10
    os.utime(lock_dir, (old, old))

    res = updater.apply(5, installed=inst, signature=sig, download_opener=op)
    assert res["applied"] is True
    assert not lock_dir.exists()   # released again once the reclaimed run finished


def test_apply_lock_with_dead_pid_reclaimed_immediately_regardless_of_age(
        tmp_path, monkeypatch, sig_env):
    """Liveness, not elapsed time, is the PRIMARY staleness signal: a lock
    recording a CONFIRMED-DEAD pid is reclaimed right away, even with the age
    threshold set so high it would never expire on its own."""
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    monkeypatch.setattr(updater, "_APPLY_LOCK_STALE_S", 10 ** 9)
    op, sig = _signed_build("0.2.0", ["click"])

    lock_dir = home / "updates" / "apply.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text("0", encoding="utf-8")   # 0 is never a live pid

    res = updater.apply(5, installed=inst, signature=sig, download_opener=op)
    assert res["applied"] is True


def test_apply_lock_with_live_pid_never_reclaimed_even_past_stale_threshold(
        tmp_path, monkeypatch, sig_env):
    """A lock recording a LIVE pid is never reclaimed, even with the age
    threshold set so low it would expire almost instantly on age alone.
    download() has no cap on total duration, only a per-socket-op timeout, so a
    large build on a slow link can legitimately run a long time."""
    import os
    import time as _time
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    monkeypatch.setattr(updater, "_APPLY_LOCK_STALE_S", 0.01)
    from localm.bugreport import LocalmError

    lock_dir = home / "updates" / "apply.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")   # us - definitely alive
    old = _time.time() - 10
    os.utime(lock_dir, (old, old))

    called = {"n": 0}
    with pytest.raises(LocalmError, match="already being applied"):
        updater.apply(5, installed=inst, signature="x",
                      download_opener=lambda *a: called.__setitem__("n", called["n"] + 1))
    assert called["n"] == 0


def test_apply_lock_released_after_success(tmp_path, monkeypatch, sig_env):
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    op, sig = _signed_build("0.2.0", ["click"])
    updater.apply(5, installed=inst, signature=sig, download_opener=op)
    assert not (home / "updates" / "apply.lock").exists()


def test_apply_lock_released_after_early_failure(tmp_path, monkeypatch, sig_env):
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    op, sig = _signed_build("0.2.0", ["click"])
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.apply(5, installed=inst, signature="AAAA", download_opener=op)  # bad signature
    assert not (home / "updates" / "apply.lock").exists()


def test_apply_cleans_up_run_scratch_after_success(tmp_path, monkeypatch, sig_env):
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    op, sig = _signed_build("0.2.0", ["click"])
    updater.apply(5, installed=inst, signature=sig, download_opener=op)
    runs_dir = home / "updates" / "runs"
    assert not runs_dir.exists() or not any(runs_dir.iterdir())


def test_apply_preserves_run_backup_when_swap_and_rollback_both_fail(tmp_path, monkeypatch, sig_env):
    """When BOTH the swap and its own internal rollback fail, the run's backup
    survives on disk; the surfaced error message names its exact path as where
    to recover from manually."""
    home, inst = _setup_apply_env(tmp_path, monkeypatch)
    from localm import _apply_update as au

    monkeypatch.setattr(au, "apply_files", lambda *a, **k: (_ for _ in ()).throw(OSError("swap boom")))
    monkeypatch.setattr(au, "rollback", lambda *a, **k: (_ for _ in ()).throw(OSError("rollback boom")))
    op, sig = _signed_build("0.2.0", ["click"])
    with pytest.raises(RuntimeError, match="rollback also failed"):
        updater.apply(5, installed=inst, signature=sig, download_opener=op)

    runs_dir = home / "updates" / "runs"
    backups = list(runs_dir.glob("*/backup")) if runs_dir.exists() else []
    assert len(backups) == 1 and backups[0].is_dir()
    assert (backups[0] / "localm" / "__init__.py").read_text(encoding="utf-8") == "# 0.1.0"
    assert not (home / "updates" / "apply.lock").exists()   # still released despite the double failure


# ------------------------ spawn_health_watchdog ---------------------------
#
# spawn_health_watchdog() never actually runs scripts/update_watchdog.py in these
# tests - it injects a fake `popen` (matching apply()'s download_opener/runner
# convention) and asserts what would have been launched.

def _capturing_popen(calls):
    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return object()   # a stand-in Popen handle; spawn_health_watchdog ignores it
    return popen


def test_spawn_health_watchdog_builds_expected_argv_and_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    calls = []
    ok = updater.spawn_health_watchdog(
        host="127.0.0.1", port=8642, scheme="http", expect_version="0.2.0",
        popen=_capturing_popen(calls))
    assert ok is True
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == sys.executable
    assert argv[1] == str(updater._watchdog_script())
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8642"
    assert "--scheme" in argv and argv[argv.index("--scheme") + 1] == "http"
    assert ("--expect-version" in argv
            and argv[argv.index("--expect-version") + 1] == "0.2.0")
    assert ("--install-root" in argv
            and argv[argv.index("--install-root") + 1] == str(updater.repo_root()))
    assert ("--log-file" in argv
            and argv[argv.index("--log-file") + 1] == str(home / "updates" / "watchdog.log"))
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["env"]["LOCALM_HOME"] == str(home)


def test_spawn_health_watchdog_posix_uses_start_new_session(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    monkeypatch.setattr(sys, "platform", "linux")
    calls = []
    updater.spawn_health_watchdog(host="127.0.0.1", port=1, scheme="http",
                                  expect_version="1", popen=_capturing_popen(calls))
    _, kwargs = calls[0]
    assert kwargs.get("start_new_session") is True
    assert "creationflags" not in kwargs


def test_spawn_health_watchdog_windows_uses_detached_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    monkeypatch.setattr(sys, "platform", "win32")
    calls = []
    updater.spawn_health_watchdog(host="127.0.0.1", port=1, scheme="http",
                                  expect_version="1", popen=_capturing_popen(calls))
    _, kwargs = calls[0]
    assert "start_new_session" not in kwargs
    assert kwargs["creationflags"] & 0x00000008   # DETACHED_PROCESS
    assert kwargs["creationflags"] & 0x00000200   # CREATE_NEW_PROCESS_GROUP


def test_spawn_health_watchdog_never_raises_when_popen_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")

    def boom(argv, **kwargs):
        raise OSError("no processes for you")

    ok = updater.spawn_health_watchdog(host="127.0.0.1", port=1, scheme="http",
                                       expect_version="1", popen=boom)
    assert ok is False


def test_spawn_health_watchdog_missing_script_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    monkeypatch.setattr(updater, "_watchdog_script", lambda: tmp_path / "nope.py")
    calls = []
    ok = updater.spawn_health_watchdog(host="127.0.0.1", port=1, scheme="http",
                                       expect_version="1", popen=_capturing_popen(calls))
    assert ok is False
    assert calls == []
