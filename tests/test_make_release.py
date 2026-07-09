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


# --------------------------------------------------------------------------- #
#  pre-publish HEAD==origin and tag-availability gates (fake git runner)      #
# --------------------------------------------------------------------------- #

_HEAD = "a" * 40
_OTHER = "b" * 40


def _git_runner(responses):
    """Fake git runner for the release gates: `responses` is a list of
    (arg-prefix, returncode, stdout); the first prefix that matches args[:len] wins."""
    def run(args):
        for prefix, rc, out in responses:
            if tuple(args[:len(prefix)]) == tuple(prefix):
                return types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def test_head_matches_origin_passes_when_equal():
    r = _git_runner([
        (["fetch"], 0, ""),
        (["rev-parse", "HEAD"], 0, _HEAD + "\n"),
        (["rev-parse", "origin/master"], 0, _HEAD + "\n"),
    ])
    make_release._require_head_matches_origin("master", runner=r)   # must not raise


def test_head_matches_origin_blocks_when_diverged():
    r = _git_runner([
        (["fetch"], 0, ""),
        (["rev-parse", "HEAD"], 0, _HEAD + "\n"),
        (["rev-parse", "origin/master"], 0, _OTHER + "\n"),
    ])
    with pytest.raises(SystemExit, match="not origin/master"):
        make_release._require_head_matches_origin("master", runner=r)


def test_tag_available_passes_when_absent():
    r = _git_runner([
        (["ls-remote"], 0, ""),                        # no origin tag
        (["rev-parse", "-q", "--verify"], 1, ""),      # no local tag
        (["rev-parse", "HEAD"], 0, _HEAD + "\n"),
    ])
    make_release._require_tag_available("v0.1.1", runner=r)   # free -> no raise


def test_tag_available_passes_when_tag_is_at_head():
    r = _git_runner([
        (["ls-remote"], 0, f"{_HEAD}\trefs/tags/v0.1.1\n"),
        (["rev-parse", "HEAD"], 0, _HEAD + "\n"),
    ])
    make_release._require_tag_available("v0.1.1", runner=r)   # already at HEAD -> ok


def test_tag_available_blocks_when_tag_at_other_commit():
    r = _git_runner([
        (["ls-remote"], 0, f"{_OTHER}\trefs/tags/v0.1.1\n"),   # stale tag (the v0.1.1 case)
        (["rev-parse", "HEAD"], 0, _HEAD + "\n"),
    ])
    with pytest.raises(SystemExit, match="already exists"):
        make_release._require_tag_available("v0.1.1", runner=r)


def test_tag_available_peels_annotated_tag_to_commit():
    # An annotated tag emits both the tag-object line AND the peeled ^{} commit line;
    # the peeled commit must be what we compare to HEAD.
    r = _git_runner([
        (["ls-remote"], 0,
         f"{'c' * 40}\trefs/tags/v0.1.1\n{_HEAD}\trefs/tags/v0.1.1^{{}}\n"),
        (["rev-parse", "HEAD"], 0, _HEAD + "\n"),
    ])
    make_release._require_tag_available("v0.1.1", runner=r)   # peeled == HEAD -> ok


# --------------------------------------------------------------------------- #
#  release TOCTOU fix: --publish must build from the GATED commit, not disk   #
# --------------------------------------------------------------------------- #
#
# scripts/make_release.py:313-334 used to run the clean-tree/HEAD/tag/verification-
# record gates, wait (multiple minutes) on require_ci_green(), and only THEN call
# build_release.build(out, force=True) - which read every file's bytes from LIVE DISK.
# A tracked-file edit landing during the CI wait would ship in the signed artifact
# having never passed CI. The fix: capture HEAD's sha immediately after the clean-
# tree/HEAD==origin gates (before the CI wait) and pass it through as build_release.
# build(..., commit=<sha>), which reads via git archive at that pinned commit (see
# tests/test_release_manifest.py for the read-side proof). This test proves the
# ORCHESTRATION side: that --publish actually threads that pinned sha through, not
# just that build_release.build() is *capable* of pinned reads.

def test_publish_pins_the_build_to_the_verified_head_sha(tmp_path, monkeypatch):
    """Full --publish flow (every gate + git call monkeypatched, no live network/git):
    build_release.build() must be invoked with commit=<the HEAD sha resolved right
    after the clean-tree/HEAD==origin gates>, not commit=None (which would mean 'read
    from whatever is on disk once CI finishes').

    Also asserts the ORDERING, not just the final value: the sha must be captured
    (git rev-parse HEAD) BEFORE require_ci_green()'s multi-minute wait, not after -
    that is the actual point of the fix (closing the TOCTOU window CI takes to run).
    A prior version of this test only checked the final captured value, which cannot
    tell 'captured before the wait' apart from 'captured after it' since every gate
    was a stateless no-op; a reintroduced ordering bug (moving the capture below
    require_ci_green) was confirmed to still pass it. This version uses a shared
    call_order list so a regression is caught even though the final commit value
    would look identical either way."""
    pinned_sha = "f" * 40
    call_order = []

    monkeypatch.setattr(make_release, "_require_clean_tree",
                        lambda: call_order.append("clean_tree"))
    monkeypatch.setattr(make_release, "_require_head_matches_origin",
                        lambda ref: call_order.append("head_matches_origin"))

    def fake_git(args, runner=None):
        if list(args) == ["rev-parse", "HEAD"]:
            call_order.append("git_rev_parse_head")
        return types.SimpleNamespace(returncode=0, stdout=pinned_sha + "\n", stderr="")
    monkeypatch.setattr(make_release, "_git", fake_git)

    monkeypatch.setattr(make_release, "_require_tag_available",
                        lambda tag: call_order.append("tag_available"))
    monkeypatch.setattr(make_release, "_require_verification_record",
                        lambda: call_order.append("verification_record"))
    monkeypatch.setattr(make_release, "require_ci_green",
                        lambda ref: call_order.append("ci_green"))

    captured = {}

    def fake_build(out, *, force=False, commit=None):
        call_order.append("build")
        captured["commit"] = commit
        captured["force"] = force
        return ["VERSION", "pyproject.toml"]
    monkeypatch.setattr(make_release.build_release, "build", fake_build)

    def fake_sign(zip_path, key, sig_path):
        sig_path.write_text("c2ln", encoding="utf-8")
        return 0
    monkeypatch.setattr(make_release.sign_release, "_sign", fake_sign)
    monkeypatch.setattr(make_release, "_verify_against_pinned", lambda out, sig: None)
    monkeypatch.setattr(make_release, "smoke_test", lambda out: None)
    monkeypatch.setattr(make_release.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0))

    key = tmp_path / "key.pem"
    key.write_text("dummy", encoding="utf-8")
    out = tmp_path / "build.zip"

    rc = make_release.main(["--key", str(key), "--out", str(out), "--publish"])

    assert rc == 0
    assert captured["commit"] == pinned_sha, (
        "build_release.build() must receive the pinned HEAD sha, not None "
        "(None means 'read from live disk' - the TOCTOU gap)")
    assert captured["force"] is True
    assert "git_rev_parse_head" in call_order

    # The ordering property: the sha capture must happen strictly before the
    # multi-minute CI wait, and the build must happen strictly after it (CI must
    # actually run before the artifact is assembled).
    idx_capture = call_order.index("git_rev_parse_head")
    idx_ci_wait = call_order.index("ci_green")
    idx_build = call_order.index("build")
    assert idx_capture < idx_ci_wait, (
        f"the HEAD sha must be captured BEFORE the CI wait, not after "
        f"(call_order={call_order!r})")
    assert idx_ci_wait < idx_build, (
        f"the build must happen AFTER CI passes, not before "
        f"(call_order={call_order!r})")
    # And the gate ordering documented in make_release.py's own comments.
    assert call_order.index("clean_tree") < call_order.index("head_matches_origin")
    assert call_order.index("head_matches_origin") < idx_capture


def test_non_publish_build_is_not_pinned_to_a_commit(tmp_path, monkeypatch):
    """A plain (no --publish) build is local dev iteration with no CI/verification
    gates run at all, so there is no verified commit to pin to - it must keep reading
    the live working tree (commit=None), exactly as before this fix."""
    captured = {}

    def fake_build(out, *, force=False, commit=None):
        captured["commit"] = commit
        return ["VERSION", "pyproject.toml"]
    monkeypatch.setattr(make_release.build_release, "build", fake_build)

    def fake_sign(zip_path, key, sig_path):
        sig_path.write_text("c2ln", encoding="utf-8")
        return 0
    monkeypatch.setattr(make_release.sign_release, "_sign", fake_sign)
    monkeypatch.setattr(make_release, "_verify_against_pinned", lambda out, sig: None)
    monkeypatch.setattr(make_release, "smoke_test", lambda out: None)

    key = tmp_path / "key.pem"
    key.write_text("dummy", encoding="utf-8")
    out = tmp_path / "build.zip"

    rc = make_release.main(["--key", str(key), "--out", str(out)])   # no --publish

    assert rc == 0
    assert captured["commit"] is None
