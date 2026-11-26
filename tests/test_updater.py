# SPDX-License-Identifier: AGPL-3.0-or-later
"""The self-updater: configured endpoint resolution, the version check, update
classification (reboot/deps/runtime/setup), and the build download. Apply is NOT
exercised here (it has its own detached-helper tests)."""

import json
import pathlib
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from localm import _version, updater, updater_handoff
from localm.bugreport import LocalmError
from tests._exe_stub import write_exe_stub
from tests._old_client_update import old_client_update
from tests._net_config import set_config
from tests._update_signing import envelope, keyset


def _opener(payload, status=200):
    """A _proxy.request opener: (method, url, data, headers, timeout) -> (status, bytes)."""
    def op(method, url, data, headers, timeout):
        op.url = url
        op.token = headers.get("X-Localm-Token")
        return status, json.dumps(payload).encode("utf-8")
    return op


# ------------------------------- endpoint -------------------------------

def test_endpoint_reuses_bugreport_config(monkeypatch):
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "https://w.example", "bugreport_upload_token": "t"})
    assert updater.endpoint() == ("https://w.example", "t")
    assert updater.available() is True


def test_endpoint_update_override_wins(monkeypatch):
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "https://bug", "bugreport_upload_token": "bt",
        "update_url": "https://upd", "update_token": "ut"})
    assert updater.endpoint() == ("https://upd", "ut")


def test_unconfigured(monkeypatch):
    set_config(monkeypatch, lambda: {})
    assert updater.available() is False


# -------------------------------- check ---------------------------------

def test_check_reports_newer(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _opener({"ok": True, "version": "v0.2.0", "notes": "stuff",
                  "asset": {"id": 3, "name": "localm-0.2.0.zip", "size": 9}})
    res = updater.check(opener=op)
    assert res["current"] == "0.1.0"
    assert res["latest"] == "v0.2.0"
    assert res["newer"] is True
    assert res["asset"]["id"] == 3
    assert op.url.endswith("/update?client=0.1.0")


def test_check_not_newer_when_same(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    res = updater.check(opener=_opener({"version": "v0.2.0"}))
    assert res["newer"] is False


def test_check_no_releases(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    res = updater.check(opener=_opener({"ok": True, "version": None}))
    assert res["latest"] is None and res["newer"] is False


def test_check_comparable_true_for_a_real_tie(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    res = updater.check(opener=_opener({"version": "v0.2.0"}))
    assert res["newer"] is False and res["comparable"] is True


def test_check_comparable_false_for_an_unrecognized_tag(monkeypatch):
    """A tag the comparator cannot order (e.g. a "stable"/"nightly" release
    name) is reported as comparable=False rather than folded into "not
    newer"."""
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.4")
    res = updater.check(opener=_opener({"version": "nightly"}))
    assert res["newer"] is False and res["comparable"] is False


def test_check_unconfigured_raises(monkeypatch):
    set_config(monkeypatch, lambda: {})
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
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "https://w", "net_mode": "off"})
    from localm.bugreport import LocalmError
    op = _counting_opener({"ok": True, "version": "v9.9.9"})
    with pytest.raises(LocalmError) as ei:
        updater.check(opener=op)
    assert op.calls["n"] == 0, "a blocked check must not attempt any connection"
    msg = f"{ei.value.summary} {ei.value.reason}".lower()
    assert "network access" in msg and "off" in msg
    from localm import config
    assert ei.value.reason == config.NET_OFF_UPDATE_REMEDY
    assert ei.value.hint == config.net_off_update_refusal("no update check was made")
    assert "settings > system > updates" in msg
    assert "localm config update_ignore_net_policy true" in msg
    assert "localm config net_mode ask" in msg


def test_check_net_mode_off_but_exempted_still_calls_opener(monkeypatch):
    """The admin-only toggle lets THIS channel through even with net_mode=off -
    the opener IS invoked, and the result is a normal successful check (not the
    blocked path)."""
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "https://w", "net_mode": "off",
        "update_ignore_net_policy": True})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _counting_opener({"ok": True, "version": "v0.2.0"})
    res = updater.check(opener=op)
    assert op.calls["n"] == 1
    assert res["newer"] is True
    assert op.url.endswith("/update?client=0.1.0")


@pytest.mark.parametrize("mode", ["ask", "allow"])
def test_check_ask_and_allow_modes_are_unaffected(monkeypatch, mode):
    """The gate fires only on the literal 'off' kill switch, so it is a no-op
    for the default 'ask' mode and for 'allow'."""
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    set_config(monkeypatch, lambda: {
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
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "https://w", "net_mode": "ask"})
    from localm.bugreport import LocalmError
    op = _counting_opener({"ok": True, "version": "v0.2.0"})
    with pytest.raises(LocalmError):
        updater.check(opener=op)
    assert op.calls["n"] == 0


def test_net_policy_allows_update_check_fails_safe_on_unreadable_config(monkeypatch, tmp_path):
    """A config.json that exists and cannot be read resolves to BLOCKED, never
    exempted, even when it held an explicit exemption."""
    import json
    import localm.config as cfg
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    home = tmp_path / ".localm"
    home.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    (home / "config.json").write_text(
        json.dumps({"net_mode": "ask", "update_ignore_net_policy": True}),
        encoding="utf-8")
    assert updater._net_policy_allows_update_check() is True
    (home / "config.json").write_bytes(bytes([0xFF, 0xFE, 0x00, 0x20]))
    (home / "config.json.bak").write_bytes(bytes([0xFF, 0xFE, 0x00, 0x20]))
    assert updater._net_policy_allows_update_check() is False


# ------------------------ prerelease channel -----------------------------

def test_check_stable_by_default_no_channel_param(monkeypatch):
    """Opt-in only: with the setting absent OR explicitly False, the request
    carries no channel parameter, only the client version."""
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _opener({"ok": True, "version": "v0.2.0"})
    updater.check(opener=op)
    assert op.url.endswith("/update?client=0.1.0")
    assert "channel" not in op.url


def test_check_prerelease_opt_in_adds_channel_param(monkeypatch):
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "https://w", "update_allow_prerelease": True})
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")
    op = _opener({"ok": True, "version": "v0.2.0"})
    updater.check(opener=op)
    assert op.url.endswith("/update?client=0.1.0&channel=prerelease")


def test_check_names_the_running_version(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "0.3.2rc1+local")
    op = _opener({"ok": True, "version": "v0.3.2"})
    updater.check(opener=op)
    assert op.url.endswith("/update?client=0.3.2rc1%2Blocal")


def test_check_without_a_readable_version_still_identifies_as_a_client(monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr(_version, "read_version", lambda: "")
    op = _opener({"ok": True, "version": "v0.3.2"})
    updater.check(opener=op)
    assert op.url.endswith("/update?client=unknown")


def test_prerelease_channel_enabled_reads_the_setting(monkeypatch):
    set_config(monkeypatch, lambda: {"update_allow_prerelease": True})
    assert updater._prerelease_channel_enabled() is True
    set_config(monkeypatch, lambda: {"update_allow_prerelease": False})
    assert updater._prerelease_channel_enabled() is False
    set_config(monkeypatch, lambda: {})
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
    set_config(monkeypatch, lambda: {
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
    attacker-controlled input to an update, via a compromised or MITM'd proxy
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


@pytest.mark.parametrize("installed", ["unknown", "", "abc"])
def test_refuse_downgrade_refuses_when_the_installed_version_is_unreadable(
        monkeypatch, installed):
    """With no readable installed version nothing can be confirmed as newer,
    so a plain update is refused whatever version the build carries."""
    from localm.bugreport import LocalmError
    monkeypatch.setattr(_version, "read_version", lambda: installed)
    with pytest.raises(LocalmError, match="installed localm version"):
        updater._refuse_downgrade("0.0.1")
    with pytest.raises(LocalmError, match="installed localm version"):
        updater._refuse_downgrade("9.9.9")


# ------------------------------ download --------------------------------

def test_download_streams_to_file(tmp_path, monkeypatch):
    set_config(monkeypatch, lambda: {
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
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.download("not-a-number", "/tmp/x")


def test_download_refuses_non_https_endpoint(tmp_path, monkeypatch):
    """A code-update download over the real urllib path refuses a non-HTTPS
    endpoint."""
    set_config(monkeypatch, lambda: {
        "bugreport_upload_url": "http://insecure.example"})   # http, not https
    from localm.bugreport import LocalmError
    with pytest.raises(LocalmError):
        updater.download(7, tmp_path / "build.zip")           # opener=None -> urllib path


# ------------------------- prepare / rollback -----------------------------
#
# An update verifies an Ed25519 signature over the downloaded build against a pinned
# public key BEFORE extracting it, and refuses a non-newer build (anti-rollback).
# These tests pin a THROWAWAY keypair and sign the fixture build, so they exercise
# the real gate rather than disabling it.

_TEST_PRIV = Ed25519PrivateKey.generate()
_TEST_RECOVERY = [Ed25519PrivateKey.generate() for _ in range(3)]


def _pyproject_str(deps):
    return ('[project]\nname = "localm"\ndependencies = ['
            + ", ".join(f'"{d}"' for d in deps) + ']\n')


def _signed_build(version, deps):
    """A deterministic build zip (bytes) + its signature envelope under the test
    release key. Returns ``(download_opener, signature)``; the opener writes the
    exact bytes that were signed, so the gate verifies them."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", version)
        z.writestr("pyproject.toml", _pyproject_str(deps))
        z.writestr("uv.lock", "version = 1\n")
        z.writestr("localm/__init__.py", f"# {version}")
    data = buf.getvalue()
    sig = envelope(data, [_TEST_PRIV])

    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return opener, sig


@pytest.fixture
def sig_env(monkeypatch):
    """Pin the test signing key and an OLDER running version so an update's signature
    gate and anti-rollback accept the 0.2.0 fixture builds."""
    monkeypatch.setattr(updater, "_PINNED_KEYSET",
                        keyset(1, [_TEST_PRIV], _TEST_RECOVERY))
    monkeypatch.setattr(_version, "read_version", lambda: "0.1.0")


def _fake_install(tmp_path, deps=("click",)):
    inst = tmp_path / "inst"
    (inst / "localm").mkdir(parents=True)
    (inst / "localm" / "__init__.py").write_text("# 0.1.0", encoding="utf-8")
    (inst / "VERSION").write_text("0.1.0", encoding="utf-8")
    (inst / "pyproject.toml").write_text(_pyproject_str(deps), encoding="utf-8")
    return inst


_UV_NAME = "uv.exe" if sys.platform == "win32" else "uv"


def _fake_uv(directory):
    """A uv program file in *directory*; returns its path."""
    exe = write_exe_stub(directory / _UV_NAME)
    exe.chmod(0o755)
    return exe


@pytest.fixture
def no_uv(tmp_path, monkeypatch):
    """A machine with no uv on PATH, none in the home directory and none bundled
    in the running environment."""
    from localm import uv_tool
    monkeypatch.setattr(uv_tool, "bundled_uv", lambda: None)
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("PATH", str(tmp_path / "emptypath"))
    return home


def _config(monkeypatch, tmp_path):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")


def test_apply_under_net_mode_off_is_refused_before_anything_is_downloaded(
        tmp_path, monkeypatch, sig_env, no_uv):
    _config(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    inst = _fake_install(tmp_path, deps=("click",))
    _fake_uv(inst / ".uv")
    ran = []
    downloads = []
    op, sig = _signed_build("0.2.0", ["click", "httpx"])
    with pytest.raises(LocalmError) as ei:
        updater_handoff.prepare(5, installed=inst, signature=sig,
                                download_opener=lambda *a: downloads.append(a) or op(*a))
    assert ei.value.stage == "network_off"
    assert downloads == []
    assert ran == [] and (inst / "VERSION").read_text().strip() == "0.1.0"


def test_deps_env_points_a_contained_install_at_its_own_cache(tmp_path, monkeypatch):
    inst = tmp_path / "inst"
    inst.mkdir()
    (inst / ".localm-install.json").write_text(json.dumps({
        "schema": 3, "runtime_contained": True,
        "cache_dir": str(inst / ".cache"), "python_dir": str(inst / ".python")}), encoding="utf-8")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "kept-as-set")
    env = updater._deps_env(inst)
    assert env["UV_CACHE_DIR"] == str(inst / ".cache")
    assert env["UV_PYTHON_INSTALL_DIR"] == "kept-as-set"
    (inst / ".localm-install.json").write_text(json.dumps({"schema": 3}), encoding="utf-8")
    assert "UV_CACHE_DIR" not in updater._deps_env(inst)


def test_rollback_last_restores(tmp_path, monkeypatch, sig_env):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    op, _sig = _signed_build("0.2.0", ["click"])
    op(None, None, None, tmp_path / "build.zip")
    old_client_update(inst, (tmp_path / "build.zip").read_bytes())
    assert (inst / "VERSION").read_text().strip() == "0.2.0"
    updater.rollback_last(installed=inst)
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    assert (inst / "localm" / "__init__.py").read_text() == "# 0.1.0"


def test_rollback_last_without_backup_raises(tmp_path, monkeypatch):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
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


def test_rollback_last_reports_a_failed_restore_as_a_localm_error(tmp_path, monkeypatch):
    install = _stage_rollback(tmp_path, monkeypatch,
                              manifest=["existing.txt", "brand_new"])

    def refuse(*a, **k):
        raise RuntimeError("refused unsafe name from the update manifest: '..'")
    monkeypatch.setattr(updater, "_restore_all", refuse)
    with pytest.raises(LocalmError) as ei:
        updater.rollback_last(installed=install)
    assert ei.value.summary.startswith("the rollback did not finish")
    assert ei.value.stage == updater.ROLLBACK_INCOMPLETE
    assert "refused unsafe name" in ei.value.reason
    assert "backup is kept at" in ei.value.reason


def test_setup_script_names_this_platforms_script(monkeypatch):
    monkeypatch.setattr(updater.sys, "platform", "linux")
    assert updater.setup_script() == "setup.sh"
    monkeypatch.setattr(updater.sys, "platform", "win32")
    assert updater.setup_script() == "setup.bat"


@pytest.mark.parametrize("what, call", [
    ("update", lambda: updater_handoff.prepare(5, signature="x")),
    ("rollback", lambda: updater.rollback_last()),
])
def test_the_container_image_refuses_an_update_and_a_rollback_before_anything(
        tmp_path, monkeypatch, what, call):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    downloads = []
    monkeypatch.setattr(updater, "download", lambda *a, **k: downloads.append(a))
    monkeypatch.setenv(updater.CONTAINER_IMAGE_ENV, "1")
    with pytest.raises(LocalmError) as ei:
        call()
    assert ei.value.summary.startswith(
        f"this localm runs from its container image, so no {what} was made")
    assert "pull the newer localm image" in ei.value.reason
    assert downloads == []
    assert not (home / "updates").exists(), "nothing was written before the refusal"


def test_the_container_marker_is_set_by_the_image_and_read_exactly():
    dockerfile = pathlib.Path(updater.__file__).resolve().parents[1] / "docker" / "Dockerfile"
    assert f"{updater.CONTAINER_IMAGE_ENV}=1" in dockerfile.read_text(encoding="utf-8")


@pytest.mark.parametrize("value, expected", [("1", True), (" 1 ", True), ("", False),
                                             ("0", False), ("yes", False)])
def test_in_container_image_reads_the_marker(monkeypatch, value, expected):
    monkeypatch.setenv(updater.CONTAINER_IMAGE_ENV, value)
    assert updater.in_container_image() is expected


def test_a_signed_archive_with_an_unsafe_path_is_a_named_refusal(tmp_path, monkeypatch, sig_env):
    import io
    import zipfile
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("VERSION", "0.2.0")
        z.writestr("pyproject.toml", _pyproject_str(["click"]))
        z.writestr("../escape.txt", "x")
    data = buf.getvalue()

    def opener(url, headers, timeout, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    with pytest.raises(LocalmError) as ei:
        updater_handoff.prepare(5, installed=inst, signature=envelope(data, [_TEST_PRIV]),
                                download_opener=opener)
    assert ei.value.summary == "refusing to install: the download is not a localm build"
    assert "unsafe path" in ei.value.reason
    assert (inst / "VERSION").read_text().strip() == "0.1.0"
    assert not (tmp_path / "escape.txt").exists()


def test_rollback_last_falls_back_to_backup_listing_without_manifest(tmp_path, monkeypatch):
    # No manifest (an older backup): backed-up entries still restore correctly; the
    # brand-new entry is not known to the fallback and survives (documented limitation).
    install = _stage_rollback(tmp_path, monkeypatch, manifest=None)
    updater.rollback_last(installed=install)
    assert (install / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"  # restored
    assert (install / "brand_new").exists()   # fallback cannot remove an unrecorded new entry


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


@pytest.mark.parametrize("doc", ["[" * 100_000, "9" * 5_000], ids=["deep", "bigint"])
def test_rollback_last_treats_a_hostile_manifest_as_unreadable(
        doc, tmp_path, monkeypatch, caplog):
    import logging
    install = _stage_rollback(tmp_path, monkeypatch, manifest=None)
    updir = (tmp_path / "home") / "updates"
    (updir / "applied_names.json").write_text(doc, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="localm"):
        updater.rollback_last(installed=install)

    assert "unreadable" in caplog.text
    assert (install / "existing.txt").read_text(encoding="utf-8") == "OLD-preapply"
    assert (install / "brand_new").exists()


def test_is_site_packages_install_recognises_a_pip_install():
    import sysconfig

    purelib = sysconfig.get_paths()["purelib"]
    assert updater.is_site_packages_install(purelib) is True
    # A self-contained release tree or source checkout is NOT one.
    assert updater.is_site_packages_install(pathlib.Path(__file__).parent) is False


def test_an_update_refuses_to_replace_a_release_tree_in_site_packages(monkeypatch, tmp_path):
    # An update replaces the release tree at repo_root(). On a pip install that
    # root is site-packages, shared with every other installed package, so the
    # update would overwrite unrelated packages and leave pip unaware of what is
    # on disk. It must refuse BEFORE downloading anything.
    import sysconfig
    from localm.bugreport import LocalmError

    purelib = sysconfig.get_paths()["purelib"]
    monkeypatch.setattr(updater, "repo_root", lambda: pathlib.Path(purelib))

    downloads = []
    monkeypatch.setattr(updater, "download",
                        lambda *a, **k: downloads.append(a))

    # Catch EVERYTHING: without the guard prepare() runs on and dies inside the
    # stubbed download with an unrelated error, which would escape a narrow
    # except and stop the data assertion below from ever speaking.
    exc = None
    try:
        updater_handoff.prepare("asset-1", signature="sig")
    except Exception as e:
        exc = e
    # DATA first: nothing was fetched, so nothing could be swapped.
    assert downloads == [], f"prepare() downloaded despite a pip install: {downloads}"
    assert exc is not None, "prepare() did not refuse a site-packages install"
    assert isinstance(exc, LocalmError), f"refused with the wrong error: {exc!r}"
    assert "pip" in str(exc).lower()


def test_an_update_refuses_a_build_that_changes_on_disk_after_the_signature_check(
        tmp_path, monkeypatch, sig_env):
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    monkeypatch.setattr("localm.config.home_dir", lambda: tmp_path / "home")
    inst = _fake_install(tmp_path, deps=("click",))
    op, sig = _signed_build("0.2.0", ["click"])
    written = {}

    def _opener(url, headers, timeout, dest):
        op(url, headers, timeout, dest)
        written["dest"] = dest

    real_verify = updater.verify_signature

    def _verify_then_tamper(data, signature):
        result = real_verify(data, signature)
        written["dest"].write_bytes(b"not the signed bytes")
        return result

    monkeypatch.setattr(updater, "verify_signature", _verify_then_tamper)
    extracted = []
    from localm import _apply_update as au
    monkeypatch.setattr(au, "extract",
                        lambda *a, **k: extracted.append(a) or None)
    with pytest.raises(LocalmError) as e:
        updater_handoff.prepare(5, installed=inst, signature=sig, download_opener=_opener)
    assert "changed after its signature was checked" in str(e.value)
    assert extracted == []
    assert (inst / "VERSION").read_text().strip() != "0.2.0"


# ----------------------- the management log line of an update -------------------

def _result(home, **fields):
    from localm.update_mechanism import applier
    (home / "updates").mkdir(parents=True, exist_ok=True)
    (home / "updates" / applier.RESULT).write_text(json.dumps(
        {"state": "new", "from": "0.1.0", "to": "0.2.0", "version": "0.2.0", "note": "",
         "notes": [], "lines": [], "at": 100.0, **fields}), encoding="utf-8")


@pytest.fixture
def full_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "full")
    set_config(monkeypatch, lambda: {"bugreport_upload_url": "https://w"})
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    monkeypatch.setattr("localm.config.HOME_DIR", home)
    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    return home


def _events(home) -> list:
    path = home / "logs" / "management.jsonl"
    return ([json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            if path.exists() else [])


def test_the_version_an_update_installed_records_it_once(full_home):
    _result(full_home)
    assert updater.log_applied_update(full_home) is True
    assert updater.log_applied_update(full_home) is False, "recorded once"
    (ev,) = _events(full_home)
    assert ev["kind"] == "update_applied"
    assert (ev["version_from"], ev["version_to"]) == ("0.1.0", "0.2.0")


@pytest.mark.parametrize("fields", [{"state": "old"}, {"to": "0.3.0"}])
def test_an_update_that_did_not_install_the_running_version_is_not_recorded(full_home,
                                                                            fields):
    _result(full_home, **fields)
    assert updater.log_applied_update(full_home) is False
    assert _events(full_home) == []


def test_a_later_update_to_the_same_version_is_recorded_again(full_home):
    _result(full_home)
    assert updater.log_applied_update(full_home) is True
    _result(full_home, at=10.0 ** 11)
    assert updater.log_applied_update(full_home) is True
    assert len(_events(full_home)) == 2


def test_privacy_mode_records_no_update(full_home, monkeypatch):
    monkeypatch.setenv("LOCALM_MODE", "privacy")
    _result(full_home)
    assert updater.log_applied_update(full_home) is False
    assert not (full_home / "logs").exists()


def test_every_start_records_the_update_it_finds(full_home, monkeypatch):
    from localm.cli import _core
    _result(full_home)
    monkeypatch.delenv("LOCALM_UPDATE_ID", raising=False)
    _core._log_applied_update("serve")
    _core._log_applied_update("_update-probe")
    assert len(_events(full_home)) == 1
    monkeypatch.setenv("LOCALM_UPDATE_ID", "0" * 32)
    _result(full_home, at=10.0 ** 11)
    _core._log_applied_update("serve")
    assert len(_events(full_home)) == 1, "a process an update started records nothing"


# ------------------- the lock of a rollback and a repair ------------------------

@pytest.fixture
def lock_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr("localm.config.home_dir", lambda: home)
    return home


def test_a_second_holder_of_the_lock_is_refused_at_once(lock_home):
    from localm.bugreport import LocalmError
    with updater._apply_lock():
        with pytest.raises(LocalmError, match="already being applied"):
            with updater._apply_lock():
                pass
    assert not (lock_home / "updates" / "apply.lock").exists()


def test_the_lock_is_given_up_after_a_failure_inside_it(lock_home):
    with pytest.raises(RuntimeError):
        with updater._apply_lock():
            raise RuntimeError("boom")
    assert not (lock_home / "updates" / "apply.lock").exists()


def test_a_fresh_lock_without_a_pid_is_not_reclaimed(lock_home):
    from localm.bugreport import LocalmError
    (lock_home / "updates" / "apply.lock").mkdir(parents=True)
    with pytest.raises(LocalmError, match="already being applied"):
        with updater._apply_lock():
            pass


def test_an_old_lock_without_a_pid_is_reclaimed(lock_home, monkeypatch):
    import os
    import time as _time
    monkeypatch.setattr(updater, "_APPLY_LOCK_STALE_S", 0.01)
    lock_dir = lock_home / "updates" / "apply.lock"
    lock_dir.mkdir(parents=True)
    old = _time.time() - 10
    os.utime(lock_dir, (old, old))
    with updater._apply_lock():
        assert (lock_dir / "pid").read_text(encoding="utf-8") == str(os.getpid())
    assert not lock_dir.exists()


def test_a_lock_of_a_dead_pid_is_reclaimed_whatever_its_age(lock_home, monkeypatch):
    monkeypatch.setattr(updater, "_APPLY_LOCK_STALE_S", 10 ** 9)
    lock_dir = lock_home / "updates" / "apply.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text("0", encoding="utf-8")   # 0 is never a live pid
    with updater._apply_lock():
        pass


def test_a_lock_of_a_running_pid_is_never_reclaimed(lock_home, monkeypatch):
    import os
    import time as _time
    from localm.bugreport import LocalmError
    monkeypatch.setattr(updater, "_APPLY_LOCK_STALE_S", 0.01)
    lock_dir = lock_home / "updates" / "apply.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
    old = _time.time() - 10
    os.utime(lock_dir, (old, old))
    with pytest.raises(LocalmError, match="already being applied"):
        with updater._apply_lock():
            pass
