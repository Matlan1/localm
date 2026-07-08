# SPDX-License-Identifier: AGPL-3.0-or-later
"""The release smoke gate (scripts/make_release.py smoke_test): a built release must
IMPORT AND RUN on its own, so a runtime-needed file omitted from the build (mis-
classified dev-only, or gitignored) is caught at build time, not by a user later.

The manifest gate proves every file is CLASSIFIED; this proves the included set is
actually COMPLETE. These are integration tests (they build a real zip and spawn a
subprocess), so they run when invoked directly / locally; the gate itself runs in
scripts/make_release.py on every release cut.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("check_manifest")
build_release = _load("build_release")
release_verify = _load("release_verify")
make_release = _load("make_release")


def _rebuild_without(src_zip: Path, dst_zip: Path, drop: str) -> None:
    """Copy *src_zip* to *dst_zip* omitting the member *drop* - simulates a release
    that lost one file to a mis-classification."""
    with zipfile.ZipFile(src_zip) as zin, zipfile.ZipFile(dst_zip, "w") as zout:
        for item in zin.infolist():
            if item.filename != drop:
                zout.writestr(item, zin.read(item.filename))


@pytest.mark.integration
class TestReleaseSmokeGate:
    def test_real_build_imports_and_runs(self, tmp_path):
        """The actual release built from the tree passes the smoke gate."""
        out = tmp_path / "build.zip"
        build_release.build(out)
        make_release.smoke_test(out)   # must not raise

    def test_catches_missing_runtime_module(self, tmp_path):
        """Drop a runtime .py the server needs -> the import check fails the gate."""
        out = tmp_path / "build.zip"
        build_release.build(out)
        assert "localm/inference/http_server.py" in set(zipfile.ZipFile(out).namelist())
        stripped = tmp_path / "stripped.zip"
        _rebuild_without(out, stripped, "localm/inference/http_server.py")
        with pytest.raises(SystemExit, match="release smoke FAILED"):
            make_release.smoke_test(stripped)

    def test_catches_missing_critical_asset(self, tmp_path):
        """Drop a critical non-.py asset -> the asset spot-check fails the gate."""
        out = tmp_path / "build.zip"
        build_release.build(out)
        stripped = tmp_path / "stripped.zip"
        _rebuild_without(out, stripped, "localm/plugins/gui/static/index.html")
        with pytest.raises(SystemExit, match="missing critical file"):
            make_release.smoke_test(stripped)


# --------------------------------------------------------------------------- #
#  pre-publish CI gate (require_ci_green) - injected gh runner, NO live CI     #
# --------------------------------------------------------------------------- #

class _FakeGh:
    """A scripted `gh` runner for require_ci_green: records calls and returns canned
    results per subcommand, with no network. `run list` returns empty on the first
    (snapshot) call, then a new run id, so the 'wait for the run to appear' loop ends."""
    def __init__(self, *, state="active", watch_rc=0, appear=True):
        self.state, self.watch_rc, self.appear = state, watch_rc, appear
        self.calls, self._lists = [], 0

    def __call__(self, args):
        self.calls.append(list(args))
        def res(rc=0, out="", err=""):
            return types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)
        if args[:1] == ["api"]:
            return res(out=json.dumps({"workflows": [
                {"path": ".github/workflows/ci.yml", "name": "CI", "state": self.state, "id": 1}]}))
        if args[:2] == ["workflow", "enable"]:
            return res()
        if args[:2] == ["workflow", "run"]:
            return res()
        if args[:2] == ["run", "list"]:
            self._lists += 1
            if self._lists == 1:
                return res(out="[]")                       # snapshot: no runs yet
            return res(out=json.dumps([{"databaseId": 999}] if self.appear else []))
        if args[:2] == ["run", "watch"]:
            return res(rc=self.watch_rc)
        return res()

    def called(self, *prefix):
        return any(c[:len(prefix)] == list(prefix) for c in self.calls)


def test_ci_gate_passes_on_active_green_ci(monkeypatch):
    monkeypatch.setattr(make_release.shutil, "which", lambda _x: "gh")
    fake = _FakeGh(state="active", watch_rc=0)
    make_release.require_ci_green("master", runner=fake, sleeper=lambda _s: None)
    assert not fake.called("workflow", "enable")           # already active -> not re-enabled
    assert fake.called("workflow", "run")                  # a run was started
    assert fake.called("run", "watch")                     # and waited on to completion


def test_ci_gate_enables_disabled_workflow_once(monkeypatch):
    monkeypatch.setattr(make_release.shutil, "which", lambda _x: "gh")
    fake = _FakeGh(state="disabled_manually", watch_rc=0)
    make_release.require_ci_green("master", runner=fake, sleeper=lambda _s: None)
    assert fake.called("workflow", "enable")               # disabled -> enabled first
    assert fake.called("workflow", "run")


def test_ci_gate_blocks_publish_on_red_ci(monkeypatch):
    monkeypatch.setattr(make_release.shutil, "which", lambda _x: "gh")
    fake = _FakeGh(state="active", watch_rc=1)              # CI run fails
    with pytest.raises(SystemExit, match="did NOT pass"):
        make_release.require_ci_green("master", runner=fake, sleeper=lambda _s: None)


def test_ci_gate_requires_gh(monkeypatch):
    monkeypatch.setattr(make_release.shutil, "which", lambda _x: None)
    with pytest.raises(SystemExit, match="gh CLI"):
        make_release.require_ci_green("master")


def test_ci_gate_errors_if_run_never_appears(monkeypatch):
    monkeypatch.setattr(make_release.shutil, "which", lambda _x: "gh")
    fake = _FakeGh(state="active", appear=False)            # triggered run never registers
    with pytest.raises(SystemExit, match="did not appear"):
        make_release.require_ci_green("master", runner=fake, sleeper=lambda _s: None,
                                      appear_timeout_s=30, poll_s=10)


# --------------------------------------------------------------------------- #
#  pre-publish live-verification record gate                                  #
# --------------------------------------------------------------------------- #

def _write_record(root, sha, verdict):
    d = root / "dev-notes" / "release-verify"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sha}.md").write_text(f"# Release verification\nVERDICT: {verdict}\n", encoding="utf-8")


def test_record_read_verdict_pass_fail_missing(tmp_path):
    _write_record(tmp_path, "abc", "PASS")
    _write_record(tmp_path, "def", "FAIL")
    assert release_verify.read_verdict("abc", tmp_path) == "PASS"
    assert release_verify.has_passing_record("abc", tmp_path)
    assert release_verify.read_verdict("def", tmp_path) == "FAIL"
    assert not release_verify.has_passing_record("def", tmp_path)
    assert release_verify.read_verdict("missing", tmp_path) is None
    assert not release_verify.has_passing_record("missing", tmp_path)


def test_publish_blocks_without_passing_record(monkeypatch):
    monkeypatch.setattr(release_verify, "current_sha", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(release_verify, "has_passing_record", lambda *a, **k: False)
    with pytest.raises(SystemExit, match="no PASSING functional-verification record"):
        make_release._require_verification_record()


def test_publish_allows_with_passing_record(monkeypatch):
    monkeypatch.setattr(release_verify, "current_sha", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(release_verify, "has_passing_record", lambda *a, **k: True)
    make_release._require_verification_record()   # must not raise


def test_changed_areas_empty_when_no_diff():
    # HEAD..HEAD has no changes -> empty list. Deterministic; exercises the git parse.
    assert release_verify.changed_areas("HEAD", REPO_ROOT) == []
