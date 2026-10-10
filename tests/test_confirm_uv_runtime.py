# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/confirm_uv_runtime.py: does a uv release work with localm.

The checks run for real against a scripted world: a fake GitHub release served
through the injected opener, and a fake ``uv`` driven through the injected
process runner. What is under test is the script's own judgement: which outcome
each observation maps to (PASS, FAIL, INCONCLUSIVE), that the installer is never
run on an unverified download, that the run is contained to its workdir, that
the receipt is complete and atomic, and that an unexpected crash is never a FAIL.

Windows adds two checks against the real thing: the real PowerShell is run on a
stub installer to prove the contained environment reaches the child, and the real
process runner is shown to kill a whole process tree on timeout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CONFIRM = _ROOT / "scripts" / "confirm_uv_runtime.py"
_BUMP = _ROOT / "scripts" / "bump_uv_pin.py"
_needs_windows = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell only")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cm():
    return _load(_CONFIRM, "confirm_uv_runtime")


@pytest.fixture(scope="module")
def bump():
    return _load(_BUMP, "bump_uv_pin")


TAG = "9.9.9"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
#  The scripted world                                                          #
# --------------------------------------------------------------------------- #

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class World:
    """A fake release, a fake machine and the fake uv that runs on it."""

    def __init__(self, tmp_path: Path, cm, bump):
        self.cm, self.bump = cm, bump
        self.tmp = tmp_path
        self.tag = TAG
        self.sh = b"#!/bin/sh\necho fake installer\n"
        self.ps1 = b"Write-Host fake installer\n"
        self.linux = b"fake tarball"
        self.uv_bytes = b"fake uv binary"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("uv-x86_64-pc-windows-msvc/uv.exe", self.uv_bytes)
        self.zip = buf.getvalue()
        self.served = {}
        self.listing_ok = True
        self.api_error = None
        self.api_overrides = {}
        self.calls = []
        self.opened = []
        self.installer_rc = 0
        self.installer_out = "installing"
        self.installs_binary = True
        self.uv_version_out = f"uv {TAG} (abc123 2026-01-01 x86_64)"
        self.python_version = "3.12.4"
        self.python_base_outside = False
        self.venv_rc = 0
        self.venv_out = ""
        self.pip_rc = 0
        self.pip_out = ""
        self.six_version = "1.17.0"
        self.lock_rc = 0
        self.lock_out = "Resolved 400 packages in 1.2s"
        self.lock_mutates = False
        self.runner_raises = None
        self.timeout_on = None
        self.root = tmp_path / "repo"
        self.build_repo()

    @property
    def digests(self) -> dict:
        return {"uv-installer.sh": _sha(self.sh), "uv-installer.ps1": _sha(self.ps1),
                "uv-x86_64-unknown-linux-gnu.tar.gz": _sha(self.linux)}

    def build_repo(self, tag: str | None = None):
        texts = {s.path: (_ROOT / s.path).read_bytes().decode("utf-8").replace("\r\n", "\n")
                 for s in self.bump.SITES}
        new = self.bump.rewrite(texts, tag or self.tag, self.digests)
        for rel, text in new.items():
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
        (self.root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    def release_body(self) -> dict:
        assets = [{"name": n, "digest": f"sha256:{d}"} for n, d in self.digests.items()]
        assets.append({"name": self.cm.ZIP_ASSET, "digest": f"sha256:{_sha(self.zip)}"})
        body = {"tag_name": self.tag, "assets": assets}
        body.update(self.api_overrides)
        return body

    def opener(self, req, timeout=None):
        url = req.full_url
        self.opened.append(url)
        if self.api_error:
            raise self.api_error
        if url.startswith("https://api.github.com/"):
            return _Resp(json.dumps(self.release_body()).encode())
        name = url.rsplit("/", 1)[1]
        data = self.served.get(name) or {
            "uv-installer.sh": self.sh, "uv-installer.ps1": self.ps1,
            self.cm.ZIP_ASSET: self.zip}.get(name)
        if data is None:
            raise OSError(f"404 {url}")
        return _Resp(data)

    def runner(self, cmd, env, cwd, timeout):
        self.calls.append((list(cmd), dict(env), cwd, timeout))
        if self.runner_raises:
            raise self.runner_raises
        cm = self.cm
        text = " ".join(str(c) for c in cmd)
        key = ("installer" if "uv-installer" in text else
               "version" if "--version" in cmd else
               "venv" if len(cmd) > 1 and cmd[1] == "venv" else
               "pip" if len(cmd) > 1 and cmd[1] == "pip" else
               "lock" if len(cmd) > 1 and cmd[1] == "lock" else
               "py_base" if "sys.base_prefix" in text else
               "py_six" if "import six" in text else "other")
        if key == self.timeout_on:
            return cm.Result(None, "", timed_out=True)
        if key == "installer":
            if self.installs_binary:
                bin_dir = Path(env["UV_UNMANAGED_INSTALL"])
                bin_dir.mkdir(parents=True, exist_ok=True)
                (bin_dir / cm.UV_NAME).write_bytes(self.uv_bytes)
            return cm.Result(self.installer_rc, self.installer_out)
        if key == "version":
            return cm.Result(0, self.uv_version_out)
        if key == "venv":
            venv = Path(cmd[-1])
            py = venv / ("Scripts/python.exe" if cm.IS_WINDOWS else "bin/python")
            if self.venv_rc == 0:
                py.parent.mkdir(parents=True, exist_ok=True)
                py.write_bytes(b"")
            return cm.Result(self.venv_rc, self.venv_out)
        if key == "pip":
            return cm.Result(self.pip_rc, self.pip_out)
        if key == "lock":
            if self.lock_mutates:
                (cwd / "uv.lock").write_text("version = 2\n", encoding="utf-8")
            return cm.Result(self.lock_rc, self.lock_out)
        if key == "py_base":
            base = (self.tmp / "elsewhere" if self.python_base_outside
                    else Path(env["UV_PYTHON_INSTALL_DIR"]) / "cpython-3.12")
            return cm.Result(0, f"{self.python_version}\n{base}\n")
        if key == "py_six":
            return cm.Result(0, f"{self.six_version}\n")
        return cm.Result(1, "unexpected command")

    def confirm(self, *, current=False, workdir=None, state_reader=None, keep=False,
                cache_dir=None):
        workdir = workdir or (self.tmp / "work")
        return self.cm.confirm(None if current else self.tag, current, workdir, cache_dir,
                               root=self.root, opener=self.opener, runner=self.runner,
                               keep=keep, base_env={"PATH": os.environ.get("PATH", ""),
                                                    "UV_INDEX_URL": "https://example.invalid",
                                                    "GITHUB_PATH": "x", "XDG_DATA_HOME": "x"},
                               state_reader=state_reader or (lambda env: {"user_bin": None}))

    def statuses(self, receipt) -> dict:
        return {n: c["status"] for n, c in receipt.checks.items()}


@pytest.fixture
def world(tmp_path, cm, bump):
    return World(tmp_path, cm, bump)


def _uv_call(world, name):
    return [c for c in world.calls if len(c[0]) > 1 and c[0][1] == name]


# --------------------------------------------------------------------------- #
#  A healthy candidate                                                         #
# --------------------------------------------------------------------------- #

def test_a_healthy_candidate_passes_every_required_check(world, cm):
    receipt = world.confirm()
    st = world.statuses(receipt)
    required = [n for n, c in receipt.checks.items() if c["required"]]
    assert set(cm.EXPECTED_REQUIRED) <= set(required)
    assert all(st[n] == "PASS" for n in required), st
    assert receipt.verdict() == "PASS" and receipt.exit_code() == 0
    assert receipt.assets == world.digests


def test_the_candidate_run_does_not_require_the_pins_to_match(world):
    receipt = world.confirm()
    check = receipt.checks["pins_match_release"]
    assert check["status"] == "SKIP" and check["required"] is False


def test_the_workdir_is_empty_again_and_the_cleanup_is_recorded(world):
    receipt = world.confirm()
    assert not (world.tmp / "work").exists()
    assert receipt.checks["cleanup"]["status"] == "PASS"
    assert receipt.checks["cleanup"]["required"] is False


def test_a_preexisting_workdir_keeps_what_was_in_it(world):
    work = world.tmp / "work"
    work.mkdir()
    (work / "mine.txt").write_text("keep", encoding="utf-8")
    receipt = world.confirm(workdir=work)
    assert receipt.verdict() == "PASS"
    assert (work / "mine.txt").read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in work.iterdir()) == ["mine.txt"]


def test_keep_leaves_the_directories_in_place(world):
    world.confirm(keep=True)
    assert (world.tmp / "work" / "uv-bin").is_dir()


def test_the_installer_and_every_uv_command_get_the_contained_environment(world, cm):
    world.confirm()
    work = (world.tmp / "work").resolve()
    for cmd, env, _cwd, _t in world.calls:
        assert "UV_INDEX_URL" not in env and "GITHUB_PATH" not in env
        assert Path(env["UV_CACHE_DIR"]).resolve().parent == work
        assert Path(env["TEMP"]).resolve().parent == work
        assert Path(env["XDG_DATA_HOME"]).resolve().parent == work
        assert env["UV_PYTHON_INSTALL_BIN"] == "0" and env["UV_PYTHON_INSTALL_REGISTRY"] == "0"
        if any("uv-installer" in str(c) for c in cmd):
            assert Path(env["UV_UNMANAGED_INSTALL"]).resolve().parent == work
            assert env["UV_NO_MODIFY_PATH"] == "1" and env["UV_DISABLE_UPDATE"] == "1"


def test_uv_runs_with_the_setup_scripts_own_flags(world):
    world.confirm()
    venv = _uv_call(world, "venv")[0][0]
    assert venv[2:6] == ["--python", "3.12", "--python-preference", "only-managed"]
    pip = _uv_call(world, "pip")[0][0]
    assert pip[2] == "install" and pip[-1] == "six==1.17.0"
    lock = _uv_call(world, "lock")[0]
    assert lock[0][2:] == ["--check"] and lock[2] == world.root


# --------------------------------------------------------------------------- #
#  The installer is never run on bytes that were not verified                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["uv-installer.ps1", "uv-installer.sh"])
def test_a_tampered_installer_fails_and_nothing_is_run(world, name):
    world.served[name] = b"echo owned"
    receipt = world.confirm()
    assert receipt.checks["installer_digests"]["status"] == "FAIL"
    assert name in receipt.checks["installer_digests"]["detail"]
    assert world.calls == []
    assert receipt.verdict() == "FAIL" and receipt.exit_code() == 1


def test_an_installer_that_cannot_be_downloaded_is_inconclusive_not_a_failure(world, monkeypatch, cm):
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    original = world.opener

    def flaky(req, timeout=None):
        if req.full_url.endswith("uv-installer.sh"):
            raise OSError("connection reset")
        return original(req, timeout)

    world.opener = flaky
    receipt = world.confirm()
    assert receipt.checks["installer_digests"]["status"] == "SKIP"
    assert receipt.verdict() == "INCONCLUSIVE" and receipt.exit_code() == 2
    assert world.calls == []


def test_the_installer_is_fetched_from_the_url_the_setup_scripts_use(world):
    world.confirm()
    for name in ("uv-installer.sh", "uv-installer.ps1"):
        assert f"https://github.com/astral-sh/uv/releases/download/{TAG}/{name}" in world.opened


# --------------------------------------------------------------------------- #
#  Outcomes                                                                    #
# --------------------------------------------------------------------------- #

def test_an_unreachable_release_api_is_inconclusive_and_runs_nothing(world):
    world.api_error = OSError("offline")
    receipt = world.confirm()
    assert receipt.checks["release_listing"]["status"] == "SKIP"
    assert receipt.verdict() == "INCONCLUSIVE"
    assert world.calls == []
    assert not (world.tmp / "work").exists()


@pytest.mark.parametrize("drop", ["uv-installer.sh", "uv-installer.ps1",
                                  "uv-x86_64-unknown-linux-gnu.tar.gz"])
def test_a_release_missing_a_pinned_asset_fails(world, drop):
    original = world.release_body

    def body():
        b = original()
        b["assets"] = [a for a in b["assets"] if a["name"] != drop]
        return b

    world.release_body = body
    receipt = world.confirm()
    assert receipt.checks["release_listing"]["status"] == "FAIL"
    assert drop in receipt.checks["release_listing"]["detail"]
    assert receipt.verdict() == "FAIL"
    assert world.calls == []


def test_a_release_listing_for_another_tag_is_inconclusive(world):
    world.api_overrides = {"tag_name": "1.0.0"}
    receipt = world.confirm()
    assert receipt.checks["release_listing"]["status"] == "SKIP"
    assert receipt.verdict() == "INCONCLUSIVE"


def test_a_failing_installer_fails(world):
    world.installer_rc, world.installer_out = 1, "error: something broke"
    receipt = world.confirm()
    assert receipt.checks["installer_run"]["status"] == "FAIL"
    assert receipt.verdict() == "FAIL"
    assert _uv_call(world, "venv") == []


def test_an_installer_that_cannot_reach_the_network_is_inconclusive(world):
    world.installer_rc = 1
    world.installer_out = "failed to download https://x: error sending request"
    receipt = world.confirm()
    assert receipt.checks["installer_run"]["status"] == "SKIP"
    assert receipt.verdict() == "INCONCLUSIVE"


def test_an_installer_that_exits_zero_without_installing_uv_fails(world):
    world.installs_binary = False
    receipt = world.confirm()
    assert receipt.checks["installer_run"]["status"] == "FAIL"
    assert receipt.verdict() == "FAIL"


def test_the_wrong_uv_version_fails(world):
    world.uv_version_out = "uv 9.9.8 (abc 2026-01-01 x86_64)"
    receipt = world.confirm()
    assert receipt.checks["version"]["status"] == "FAIL"
    assert receipt.verdict() == "FAIL"


@pytest.mark.parametrize("out", ["", "not uv", "uv 9.9.99 (x)", "uv 19.9.9 (x)"])
def test_the_version_must_match_exactly(world, out):
    world.uv_version_out = out
    assert world.confirm().checks["version"]["status"] == "FAIL"


def test_a_venv_on_the_wrong_python_fails(world):
    world.python_version = "3.11.9"
    receipt = world.confirm()
    assert receipt.checks["venv_python"]["status"] == "FAIL"
    assert "expected 3.12" in receipt.checks["venv_python"]["detail"]
    assert receipt.verdict() == "FAIL"


def test_a_python_that_is_not_the_managed_one_under_the_workdir_fails(world):
    world.python_base_outside = True
    receipt = world.confirm()
    assert receipt.checks["venv_python"]["status"] == "FAIL"
    assert "not under the managed Python directory" in receipt.checks["venv_python"]["detail"]


def test_a_failed_venv_is_a_failure_and_the_install_is_not_attempted(world):
    world.venv_rc, world.venv_out = 2, "error: No interpreter found for Python 3.12"
    receipt = world.confirm()
    assert receipt.checks["venv_python"]["status"] == "FAIL"
    assert receipt.checks["pip_install"]["status"] == "SKIP"
    assert _uv_call(world, "pip") == []
    assert receipt.verdict() == "FAIL"


def test_a_venv_download_blocked_by_the_network_is_inconclusive(world):
    world.venv_rc, world.venv_out = 2, "error: Failed to download Python: error sending request"
    receipt = world.confirm()
    assert receipt.checks["venv_python"]["status"] == "SKIP"
    assert receipt.verdict() == "INCONCLUSIVE"


def test_a_pip_install_that_leaves_the_package_unimportable_fails(world):
    world.six_version = "0.0.1"
    receipt = world.confirm()
    assert receipt.checks["pip_install"]["status"] == "FAIL"
    assert receipt.verdict() == "FAIL"


def test_a_pip_install_error_fails(world):
    world.pip_rc, world.pip_out = 2, "error: No solution found"
    assert world.confirm().checks["pip_install"]["status"] == "FAIL"


def test_a_lock_the_candidate_calls_stale_fails(world):
    world.lock_rc = 1
    world.lock_out = ("Resolved 159 packages in 9.62s\nerror: The lockfile at `uv.lock` needs to be "
                      "updated, but `--check` was provided.\n\nhint: To update the lockfile, run `uv lock`.")
    receipt = world.confirm()
    assert receipt.checks["lock_check"]["status"] == "FAIL"
    assert receipt.verdict() == "FAIL"


def test_a_lock_check_that_cannot_reach_the_index_is_inconclusive(world):
    world.lock_rc = 2
    world.lock_out = "error: Failed to fetch: error sending request for url (https://pypi.org/)"
    receipt = world.confirm()
    assert receipt.checks["lock_check"]["status"] == "SKIP"
    assert receipt.verdict() == "INCONCLUSIVE"


def test_a_lock_check_failing_for_another_reason_fails(world):
    world.lock_rc = 2
    world.lock_out = "error: Failed to parse uv.lock: unsupported version 9"
    assert world.confirm().checks["lock_check"]["status"] == "FAIL"


def test_a_lock_check_that_rewrites_the_lock_fails_and_the_lock_is_restored(world):
    world.lock_mutates = True
    receipt = world.confirm()
    assert receipt.checks["lock_check"]["status"] == "FAIL"
    assert (world.root / "uv.lock").read_text(encoding="utf-8") == "version = 1\n"
    assert receipt.verdict() == "FAIL"


@pytest.mark.parametrize("step", ["installer", "venv", "pip", "lock"])
def test_a_step_that_times_out_is_inconclusive(world, step):
    world.timeout_on = step
    receipt = world.confirm()
    assert receipt.verdict() == "INCONCLUSIVE", world.statuses(receipt)
    assert "SKIP" in world.statuses(receipt).values()


def test_an_unexpected_crash_is_inconclusive_never_a_failure(world):
    world.runner_raises = RuntimeError("boom")
    receipt = world.confirm()
    assert receipt.verdict() == "INCONCLUSIVE" and receipt.exit_code() == 2
    assert "RuntimeError: boom" in receipt.why()
    assert not (world.tmp / "work").exists()


def test_a_crash_after_every_check_passed_is_still_not_a_pass(world, cm):
    receipt = cm.Receipt(TAG, False)
    for n in cm.EXPECTED_REQUIRED:
        receipt.add(n, "PASS", "ok")
    assert receipt.verdict() == "PASS"
    receipt.note = "unexpected OSError: disk"
    assert receipt.verdict() == "INCONCLUSIVE"


def test_the_installer_is_not_run_when_isolation_cannot_be_shown(world, monkeypatch, cm):
    monkeypatch.setattr(cm, "check_env_contained", lambda env, d: ["TEMP"])
    receipt = world.confirm()
    assert receipt.checks["isolation"]["status"] == "FAIL"
    assert receipt.verdict() == "FAIL"
    assert world.calls == []


# --------------------------------------------------------------------------- #
#  Containment                                                                 #
# --------------------------------------------------------------------------- #

def test_a_change_to_the_real_machine_state_fails_the_run(world):
    states = iter([{"hkcu_path": "C:\\a", "user_bin": None},
                   {"hkcu_path": "C:\\a;C:\\uv", "user_bin": None}])
    receipt = world.confirm(state_reader=lambda env: next(states))
    assert receipt.checks["containment"]["status"] == "FAIL"
    assert "hkcu_path" in receipt.checks["containment"]["detail"]
    assert receipt.verdict() == "FAIL"


def test_unchanged_machine_state_passes_containment(world):
    receipt = world.confirm(state_reader=lambda env: {"hkcu_path": "C:\\a", "user_bin": None})
    assert receipt.checks["containment"]["status"] == "PASS"


def test_containment_is_measured_even_when_the_installer_failed(world):
    world.installer_rc = 1
    states = iter([{"hkcu_path": "a"}, {"hkcu_path": "b"}])
    receipt = world.confirm(state_reader=lambda env: next(states))
    assert receipt.checks["installer_run"]["status"] == "FAIL"
    assert receipt.checks["containment"]["status"] == "FAIL"


def test_containment_is_not_claimed_when_the_installer_was_never_started(world):
    world.served["uv-installer.sh"] = b"tampered"
    receipt = world.confirm()
    assert receipt.checks["containment"]["status"] == "SKIP"


def test_state_diff_names_each_changed_key(cm):
    assert cm.state_diff({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == ["b", "c"]
    assert cm.state_diff({"a": 1}, {"a": 1}) == []


# --------------------------------------------------------------------------- #
#  --current                                                                   #
# --------------------------------------------------------------------------- #

def test_current_confirms_the_pinned_version_read_from_setup_sh(world):
    receipt = world.confirm(current=True)
    assert receipt.tag == TAG and receipt.current is True
    assert receipt.checks["pins_match_release"]["status"] == "PASS"
    assert receipt.checks["pins_match_release"]["required"] is True
    assert receipt.verdict() == "PASS"
    assert json.loads(json.dumps(receipt.to_json()))["current"] is True


def test_current_fails_when_the_dockerfile_lags_the_installers(world, bump):
    site = next(s for s in bump.SITES if s.path == "docker/Dockerfile")
    path = world.root / "docker/Dockerfile"
    text = path.read_text(encoding="utf-8")
    path.write_text(bump._replace(site.version_re, text, "9.9.8", "v"), encoding="utf-8", newline="\n")
    receipt = world.confirm(current=True)
    check = receipt.checks["pins_match_release"]
    assert check["status"] == "FAIL" and "different uv releases" in check["detail"]
    assert "docker/Dockerfile=9.9.8" in check["detail"]
    assert receipt.verdict() == "FAIL"


@pytest.mark.parametrize("rel", ["setup.sh", "setup-gui.sh", "setup.bat", "setup-gui.bat",
                                 "docker/Dockerfile"])
def test_current_fails_when_any_one_place_pins_a_wrong_digest(world, bump, rel):
    site = next(s for s in bump.SITES if s.path == rel)
    path = world.root / rel
    text = path.read_text(encoding="utf-8")
    path.write_text(bump._replace(site.sha_re, text, "0" * 64, "sha"), encoding="utf-8",
                    newline="\n")
    receipt = world.confirm(current=True)
    assert receipt.checks["pins_match_release"]["status"] == "FAIL"
    assert rel in receipt.checks["pins_match_release"]["detail"]
    assert receipt.verdict() == "FAIL"


def test_current_with_an_unreadable_pin_fails(world):
    (world.root / "setup.sh").write_text("# no pin here\n", encoding="utf-8")
    receipt = world.confirm(current=True)
    assert receipt.verdict() == "FAIL"
    assert receipt.tag == "unknown"
    assert world.calls == []


def test_current_hashes_what_setup_would_download_against_the_pinned_value(world, bump):
    world.served["uv-installer.ps1"] = world.ps1
    receipt = world.confirm(current=True)
    assert "and to the pinned values" in receipt.checks["installer_digests"]["detail"]


# --------------------------------------------------------------------------- #
#  The release binary (Windows zip)                                            #
# --------------------------------------------------------------------------- #

def _binary_case(world, cm, monkeypatch, *, installed: bytes, cache_dir: Path | None = None):
    monkeypatch.setattr(cm, "IS_WINDOWS", True)
    monkeypatch.setattr(cm, "UV_NAME", "uv.exe")
    d = cm.Dirs(world.tmp / "w")
    d.make()
    uv = d.uv_bin / "uv.exe"
    uv.write_bytes(installed)
    r = cm.Receipt(TAG, False)
    r.listing = {cm.ZIP_ASSET: _sha(world.zip)}
    cm.check_release_binary(r, world.bump, TAG, uv, cache_dir or d.dl, d, world.opener)
    return r.checks["release_binary"]


def test_the_installed_binary_must_equal_the_one_in_the_verified_zip(world, cm, monkeypatch):
    assert _binary_case(world, cm, monkeypatch, installed=world.uv_bytes)["status"] == "PASS"
    bad = _binary_case(world, cm, monkeypatch, installed=b"swapped")
    assert bad["status"] == "FAIL" and "hashes to" in bad["detail"]


def test_a_zip_that_does_not_match_its_api_digest_fails(world, cm, monkeypatch):
    world.served[cm.ZIP_ASSET] = b"not the release zip"
    check = _binary_case(world, cm, monkeypatch, installed=world.uv_bytes)
    assert check["status"] == "FAIL" and "API publishes" in check["detail"]


def test_a_cached_zip_is_reused_only_when_its_hash_matches(world, cm, monkeypatch):
    cache = world.tmp / "cache"
    (cache / TAG).mkdir(parents=True)
    (cache / TAG / cm.ZIP_ASSET).write_bytes(b"corrupt")
    check = _binary_case(world, cm, monkeypatch, installed=world.uv_bytes, cache_dir=cache)
    assert check["status"] == "PASS"
    assert (cache / TAG / cm.ZIP_ASSET).read_bytes() == world.zip
    opened = len(world.opened)
    check = _binary_case(world, cm, monkeypatch, installed=world.uv_bytes, cache_dir=cache)
    assert check["status"] == "PASS" and len(world.opened) == opened


def test_the_binary_comparison_is_not_required_off_windows(world, cm, monkeypatch):
    monkeypatch.setattr(cm, "IS_WINDOWS", False)
    r = cm.Receipt(TAG, False)
    cm.check_release_binary(r, world.bump, TAG, Path("x"), Path("y"), None, None)
    assert r.checks["release_binary"] == {
        "status": "SKIP", "required": False, "detail": "compared against the Windows zip only"}


# --------------------------------------------------------------------------- #
#  Receipt and command line                                                    #
# --------------------------------------------------------------------------- #

def test_verdict_rules(cm):
    def c(status, required=True):
        return {"status": status, "required": required}
    assert cm.verdict_of({"a": c("PASS"), "b": c("PASS")}) == "PASS"
    assert cm.verdict_of({"a": c("PASS"), "b": c("SKIP")}) == "INCONCLUSIVE"
    assert cm.verdict_of({"a": c("FAIL"), "b": c("SKIP")}) == "FAIL"
    assert cm.verdict_of({"a": c("PASS"), "b": c("FAIL", False), "c": c("SKIP", False)}) == "PASS"
    assert cm.verdict_of({}) == "INCONCLUSIVE"
    assert cm.verdict_of({"a": c("PASS")}, expected=("a", "b")) == "INCONCLUSIVE"
    assert cm.verdict_of({"a": c("PASS", False)}) == "INCONCLUSIVE"


def test_the_expected_required_names_are_the_ones_the_bump_demands(cm, bump):
    assert tuple(cm.EXPECTED_REQUIRED) == tuple(bump.REQUIRED_CHECKS)


def test_a_passing_receipt_satisfies_the_bump_and_a_tampered_one_does_not(world, cm, bump, tmp_path):
    path = tmp_path / "r.json"
    receipt = world.confirm()
    receipt.write(path)
    assert bump.load_receipt(path, TAG) == world.digests
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checks"]["lock_check"]["status"] = "FAIL"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(bump.Refused, match="lock_check: FAIL"):
        bump.load_receipt(path, TAG)


def test_the_receipt_carries_the_contract_fields(world):
    data = world.confirm().to_json()
    assert set(data) == {"schema", "component", "tag", "current", "verdict", "why", "written_at",
                         "hardware", "checks", "assets"}
    assert data["schema"] == 1 and data["component"] == "uv"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", data["written_at"])
    for check in data["checks"].values():
        assert set(check) == {"status", "required", "detail"}
        assert check["status"] in ("PASS", "FAIL", "SKIP") and isinstance(check["required"], bool)


def test_the_receipt_is_written_atomically(cm, tmp_path):
    path = tmp_path / "deep" / "r.json"
    cm.Receipt(TAG, False).write(path)
    assert json.loads(path.read_text(encoding="utf-8"))["verdict"] == "INCONCLUSIVE"
    assert [p.name for p in path.parent.iterdir()] == ["r.json"]


def test_main_writes_a_receipt_and_exits_2_for_a_bad_tag(cm, tmp_path, capsys):
    path = tmp_path / "r.json"
    rc = cm.main(["--tag", "v1.2", "--workdir", str(tmp_path / "w"), "--receipt", str(path)])
    assert rc == 2
    assert json.loads(path.read_text(encoding="utf-8"))["verdict"] == "INCONCLUSIVE"
    assert "not a uv release version" in capsys.readouterr().out


def test_main_writes_the_receipt_even_when_the_run_is_interrupted(cm, tmp_path, monkeypatch):
    def interrupted(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cm, "confirm", interrupted)
    path = tmp_path / "r.json"
    rc = cm.main(["--tag", "1.2.3", "--workdir", str(tmp_path / "w"), "--receipt", str(path)])
    assert rc == 2
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["verdict"] == "INCONCLUSIVE" and "KeyboardInterrupt" in data["why"]


def test_main_requires_exactly_one_of_tag_and_current(cm, tmp_path):
    with pytest.raises(SystemExit):
        cm.main(["--workdir", str(tmp_path), "--receipt", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit):
        cm.main(["--tag", "1.2.3", "--current", "--workdir", str(tmp_path),
                 "--receipt", str(tmp_path / "r.json")])


# --------------------------------------------------------------------------- #
#  Small pieces                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, expected", [
    ("error sending request for url", "SKIP"), ("dns error: failed to lookup", "SKIP"),
    ("operation timed out", "SKIP"), ("error: No solution found", "FAIL"),
    ("", "FAIL"), (None, "FAIL")])
def test_classify_failure(cm, text, expected):
    assert cm.classify_failure(text) == expected


def test_contained_env_removes_inherited_uv_and_redirects_every_location(cm, tmp_path):
    d = cm.Dirs((tmp_path / "w").resolve())
    d.make()
    base = {"PATH": "p", "UV_INDEX_URL": "x", "UV_PYTHON_PREFERENCE": "system", "GITHUB_PATH": "g",
            "XDG_BIN_HOME": "b", "CARGO_DIST_FORCE_INSTALL_DIR": "c", "INSTALLER_NO_MODIFY_PATH": "1",
            "PSModulePath": "m", "HTTPS_PROXY": "http://proxy", "TEMP": "C:/real"}
    env = cm.contained_env(base, d)
    assert env["PATH"] == "p" and env["HTTPS_PROXY"] == "http://proxy"
    assert not {"GITHUB_PATH", "XDG_BIN_HOME", "CARGO_DIST_FORCE_INSTALL_DIR",
                "INSTALLER_NO_MODIFY_PATH", "PSModulePath", "UV_INDEX_URL",
                "UV_PYTHON_PREFERENCE"} & set(env)
    assert cm.check_env_contained(env, d) == []
    assert env["TEMP"] != "C:/real"


@pytest.mark.parametrize("key", ["TEMP", "UV_CACHE_DIR", "LOCALAPPDATA", "XDG_CONFIG_HOME"])
def test_check_env_contained_flags_a_location_outside_the_workdir(cm, tmp_path, key):
    d = cm.Dirs((tmp_path / "w").resolve())
    d.make()
    env = cm.contained_env({}, d)
    env[key] = str(tmp_path / "elsewhere")
    assert cm.check_env_contained(env, d) == [key]
    env = cm.contained_env({}, d)
    env["GITHUB_PATH"] = "x"
    env["UV_INDEX_URL"] = "x"
    assert cm.check_env_contained(env, d) == ["GITHUB_PATH", "UV_INDEX_URL"]


def test_installer_env_adds_only_the_unmanaged_install_dir(cm, tmp_path):
    d = cm.Dirs((tmp_path / "w").resolve())
    d.make()
    env = cm.installer_env({"PATH": "p"}, d)
    assert env["UV_UNMANAGED_INSTALL"] == str(d.uv_bin)
    assert cm.check_env_contained(env, d) == []


def test_download_retries_then_returns_the_hash_and_leaves_no_partial_file(cm, tmp_path, monkeypatch):
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    attempts = []

    def opener(req, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("reset")
        return _Resp(b"payload")

    dest = tmp_path / "sub" / "f.bin"
    assert cm.download("https://x/f.bin", dest, opener) == _sha(b"payload")
    assert dest.read_bytes() == b"payload" and len(attempts) == 3
    assert [p.name for p in dest.parent.iterdir()] == ["f.bin"]


def test_download_gives_up_after_its_attempts_and_raises(cm, tmp_path, monkeypatch):
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)

    def opener(req, timeout=None):
        raise OSError("reset")

    with pytest.raises(OSError, match="reset"):
        cm.download("https://x/f.bin", tmp_path / "f.bin", opener)
    assert list(tmp_path.iterdir()) == []


def test_rmtree_removes_read_only_files(cm, tmp_path):
    target = tmp_path / "t" / "sub"
    target.mkdir(parents=True)
    f = target / "ro.txt"
    f.write_text("x", encoding="utf-8")
    os.chmod(f, 0o400)
    assert cm.rmtree(tmp_path / "t") == []
    assert not (tmp_path / "t").exists()


# --------------------------------------------------------------------------- #
#  The real process runner and the real PowerShell                              #
# --------------------------------------------------------------------------- #

def test_the_runner_captures_output_and_the_exit_code(cm):
    res = cm.run_process([sys.executable, "-c", "print('hi'); raise SystemExit(3)"],
                         dict(os.environ), None, 60)
    assert res.rc == 3 and res.out.strip() == "hi" and not res.timed_out


def test_a_timed_out_command_is_killed_with_its_whole_process_tree(cm, tmp_path):
    psutil = pytest.importorskip("psutil")
    pidfile = tmp_path / "grandchild.pid"
    child = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open(r'{pidfile}', 'w').write(str(p.pid))\n"
        "time.sleep(120)\n")
    res = cm.run_process([sys.executable, "-c", child], dict(os.environ), None, 5)
    assert res.timed_out and res.rc is None
    grandchild = int(pidfile.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 15
    while psutil.pid_exists(grandchild) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not psutil.pid_exists(grandchild)


_STUB_PS1 = (
    "$names = 'UV_UNMANAGED_INSTALL','TEMP','TMP','GITHUB_PATH','LOCALAPPDATA','APPDATA',"
    "'UV_INDEX_URL','UV_NO_MODIFY_PATH','UV_DISABLE_UPDATE','XDG_DATA_HOME','PSModulePath'\n"
    "$out = @{}\n"
    "foreach ($n in $names) { $out[$n] = [Environment]::GetEnvironmentVariable($n) }\n"
    "$dir = $env:UV_UNMANAGED_INSTALL\n"
    "New-Item -ItemType Directory -Force -Path $dir | Out-Null\n"
    "$out | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $dir 'env.json')\n"
    "Set-Content -LiteralPath (Join-Path $dir 'uv.exe') -Value 'stub'\n")


@_needs_windows
def test_real_powershell_receives_the_contained_environment(cm, bump, tmp_path):
    d = cm.Dirs((tmp_path / "w").resolve())
    d.make()
    ps1 = d.dl / "uv-installer.ps1"
    ps1.write_text(_STUB_PS1, encoding="utf-8")
    base = dict(os.environ)
    base.update({"GITHUB_PATH": str(tmp_path / "gh.txt"), "UV_INDEX_URL": "https://example.invalid",
                 "XDG_DATA_HOME": str(tmp_path / "xdg")})
    r = cm.Receipt(TAG, False)
    uv = cm.check_installer_run(r, d, base, {bump.ASSET_PS1: ps1, bump.ASSET_SH: ps1},
                                cm.run_process, bump)
    assert r.checks["installer_run"]["status"] == "PASS", r.checks
    assert uv == d.uv_bin / "uv.exe"
    seen = json.loads((d.uv_bin / "env.json").read_text(encoding="utf-8-sig"))
    assert seen["UV_UNMANAGED_INSTALL"] == str(d.uv_bin)
    assert seen["UV_NO_MODIFY_PATH"] == "1" and seen["UV_DISABLE_UPDATE"] == "1"
    assert seen["GITHUB_PATH"] is None and seen["UV_INDEX_URL"] is None
    for key in ("TEMP", "TMP", "LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME"):
        assert Path(seen[key]).resolve().parent == d.root, key
    assert not (tmp_path / "gh.txt").exists()


@_needs_windows
def test_real_machine_state_is_readable_and_stable_across_two_reads(cm):
    first, second = cm.real_state(), cm.real_state()
    assert {"hkcu_path", "hkcu_python", "localappdata_receipt", "user_bin"} <= set(first)
    assert cm.state_diff(first, second) == []
