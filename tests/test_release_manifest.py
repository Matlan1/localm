# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the release-file manifest gate (NEW-RELEASE-FILEMANIFEST).

Two layers:
  - Pure, git-free unit tests of the classification core (_match, classify_problems,
    release_files, load_manifest, _probe_path) and of build_release's guards. These
    run everywhere, including CI's default `pytest -m "not integration"` pass, so the
    drift-detection logic is proven on synthetic trees.
  - Integration tests (@pytest.mark.integration; require git + the real checkout) that
    assert the REAL tree verifies clean, every tracked file is classified exactly once,
    each local-only pattern is genuinely git-ignored, and scripts/build_release.py
    produces a build.zip the updater's own verify_zip accepts. The gate is ALSO enforced
    in CI directly via `python scripts/check_hygiene.py` (which folds the manifest check
    in), so these being marked integration does not weaken CI enforcement.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # so build_release's `import check_manifest` reuses this one
    spec.loader.exec_module(mod)
    return mod


cm = _load("check_manifest")
build_release = _load("build_release")


# --------------------------------------------------------------------------- #
#  _match: pattern semantics                                                  #
# --------------------------------------------------------------------------- #

class TestMatch:
    def test_directory_prefix(self):
        assert cm._match("tests/", "tests/foo/bar.py")
        assert cm._match("tests/", "tests")            # the dir itself
        assert not cm._match("tests/", "tests-js/x")   # sibling, not a child
        assert not cm._match("tests/", "atests/x")

    def test_exact_path(self):
        assert cm._match("VERSION", "VERSION")
        assert not cm._match("VERSION", "VERSION.txt")
        assert not cm._match("VERSION", "sub/VERSION")

    def test_glob_basename(self):
        assert cm._match("*_signing_key.pem", "update_signing_key.pem")
        assert cm._match("*_signing_key.pem", "a/b/my_signing_key.pem")   # basename match
        assert cm._match("*.env", "prod.env")
        assert not cm._match("*.env", "env")

    def test_glob_fullpath(self):
        assert cm._match("localm/*/plug.py", "localm/rag/plug.py")


# --------------------------------------------------------------------------- #
#  classify_problems: the drift rules, on synthetic trees                     #
# --------------------------------------------------------------------------- #

class TestClassifyProblems:
    INC = ["localm/", "VERSION", "pyproject.toml"]
    EXC = ["tests/", "AGENTS.md"]
    LOC = ["issues/", "config.json", "*_signing_key.pem"]

    def _clean_tree(self):
        return ["localm/cli.py", "VERSION", "pyproject.toml", "tests/test_x.py", "AGENTS.md"]

    def test_clean_tree_has_no_problems(self):
        assert cm.classify_problems(self._clean_tree(), self.INC, self.EXC, self.LOC) == []

    def test_unclassified_file_is_flagged(self):
        tree = self._clean_tree() + ["newtool/thing.py"]
        problems = cm.classify_problems(tree, self.INC, self.EXC, self.LOC)
        assert any("newtool/thing.py" in p and "unclassified" in p for p in problems), problems

    def test_ambiguous_file_is_flagged(self):
        # A file matching BOTH an include and an exclude pattern.
        inc = self.INC + ["shared/"]
        exc = self.EXC + ["shared/"]
        tree = self._clean_tree() + ["shared/x.py"]
        problems = cm.classify_problems(tree, inc, exc, self.LOC)
        assert any("shared/x.py" in p and "BOTH" in p for p in problems), problems

    def test_local_only_leak_is_flagged(self):
        tree = self._clean_tree() + ["issues/secret.md"]
        problems = cm.classify_problems(tree, self.INC, self.EXC, self.LOC)
        assert any("issues/secret.md" in p and "local-only" in p for p in problems), problems

    def test_leaked_secret_by_glob_is_flagged(self):
        tree = self._clean_tree() + ["update_signing_key.pem"]
        problems = cm.classify_problems(tree, self.INC, self.EXC, self.LOC)
        # It is both unclassified AND a leak; the leak message must be present.
        assert any("update_signing_key.pem" in p and "local-only" in p for p in problems), problems

    @pytest.mark.parametrize(
        ("list_attr", "extra_pattern", "expect_substr"),
        [
            ("INC", "gone/", "release.include"),
            ("EXC", "removed_dir/", "release.exclude"),
        ],
    )
    def test_stale_pattern_is_flagged(self, list_attr, extra_pattern, expect_substr):
        inc = self.INC + [extra_pattern] if list_attr == "INC" else self.INC
        exc = self.EXC + [extra_pattern] if list_attr == "EXC" else self.EXC
        problems = cm.classify_problems(self._clean_tree(), inc, exc, self.LOC)
        assert any(
            extra_pattern in p and "stale" in p and expect_substr in p for p in problems
        ), problems

    def test_rename_fires_both_ends(self):
        # tests/ renamed to test/: the tests/ exclude goes stale AND the new files
        # are unclassified. Both signals must appear.
        tree = ["localm/cli.py", "VERSION", "pyproject.toml", "test/test_x.py", "AGENTS.md"]
        problems = cm.classify_problems(tree, self.INC, self.EXC, self.LOC)
        assert any("tests/" in p and "stale" in p for p in problems), problems
        assert any("test/test_x.py" in p and "unclassified" in p for p in problems), problems


# --------------------------------------------------------------------------- #
#  release_files + _probe_path                                                #
# --------------------------------------------------------------------------- #

def test_release_files_is_include_minus_exclude():
    tree = ["localm/a.py", "tests/t.py", "scripts/report_issue.py",
            "scripts/check_hygiene.py", "VERSION"]
    inc = ["localm/", "VERSION", "scripts/report_issue.py"]
    exc = ["tests/", "scripts/check_hygiene.py"]
    got = cm.release_files(tree, inc, exc)
    assert got == ["VERSION", "localm/a.py", "scripts/report_issue.py"]


def test_probe_path_forms():
    assert cm._probe_path("issues/") == "issues/__manifest_probe__"
    assert cm._probe_path("config.json") == "config.json"
    assert cm._probe_path("*_signing_key.pem") == "x_signing_key.pem"
    assert cm._probe_path("*.env") == "x.env"


def test_probe_path_bracket_class_picks_a_real_member():
    # A '[...]' class must become a member that actually satisfies it, so the
    # check-ignore cross-check does not spuriously fail on a bracket pattern.
    import fnmatch
    for pat in ("file[0-9].pem", "key[abc].env", "x[!z].env"):
        probe = cm._probe_path(pat)
        assert fnmatch.fnmatchcase(probe, pat), (pat, probe)


def test_match_is_case_sensitive_uniformly():
    # Exact, dir-prefix, and glob all match case-sensitively (git/Linux semantics),
    # so behavior does not flip between a Windows box and Linux CI.
    assert cm._match("README.md", "readme.md") is False
    assert cm._match("docs/", "DOCS/x.md") is False
    assert cm._match("*.env", "PROD.ENV") is False
    assert cm._match("*.env", "prod.env") is True


# --------------------------------------------------------------------------- #
#  load_manifest: fails loud on missing / malformed                           #
# --------------------------------------------------------------------------- #

def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cm.load_manifest(tmp_path / "nope.toml")


def test_load_manifest_malformed_raises(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text("[release]\n[release.include]\n# no patterns key\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cm.load_manifest(p)


def test_load_real_manifest_has_three_lists():
    man = cm.load_manifest()
    assert man["include"] and man["exclude"] and man["local_only"]
    # sanity: the updater-required files are declared release-include.
    assert "VERSION" in man["include"] and "pyproject.toml" in man["include"]


# --------------------------------------------------------------------------- #
#  rule-5 sensitive branches: surface real failures, do not hide them         #
# --------------------------------------------------------------------------- #

def test_check_ignore_hard_error_is_surfaced(monkeypatch):
    """A git check-ignore FATAL error (exit 128) must surface as a problem, never a
    silent pass (AGENTS.md rule 5)."""
    class _Res:
        returncode = 128
        stdout = b""
        stderr = b"fatal: bad thing"
    monkeypatch.setattr(cm.subprocess, "run", lambda *a, **k: _Res())
    problems = cm._local_ignore_problems(["issues/"])
    assert any("git check-ignore failed" in p for p in problems), problems


def test_git_absent_skips_gate_without_masking(monkeypatch):
    """When git is unavailable the gate does not apply (returns []), the one
    documented silent case - but it must not swallow a real classification drift that
    a working git would have caught. With tracked_files -> None, check_manifest returns
    [] even though the manifest itself is well-formed."""
    monkeypatch.setattr(cm, "tracked_files", lambda *a, **k: None)
    assert cm.check_manifest() == []


# --------------------------------------------------------------------------- #
#  build_release guards (git-free via monkeypatch)                            #
# --------------------------------------------------------------------------- #

def test_build_refuses_dirty_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "check_manifest", lambda *a, **k: ["some drift"])
    with pytest.raises(ValueError, match="refusing to build"):
        build_release.build(tmp_path / "build.zip")


def test_build_requires_version(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "check_manifest", lambda *a, **k: [])
    monkeypatch.setattr(cm, "load_manifest", lambda *a, **k: {
        "include": ["localm/", "pyproject.toml"], "exclude": [], "local_only": []})
    monkeypatch.setattr(cm, "tracked_files", lambda *a, **k: ["localm/a.py", "pyproject.toml"])
    with pytest.raises(ValueError, match="missing 'VERSION'"):
        build_release.build(tmp_path / "build.zip")


def test_build_refuses_existing_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "check_manifest", lambda *a, **k: [])
    monkeypatch.setattr(cm, "load_manifest", lambda *a, **k: {
        "include": ["VERSION", "pyproject.toml"], "exclude": [], "local_only": []})
    monkeypatch.setattr(cm, "tracked_files", lambda *a, **k: ["VERSION", "pyproject.toml"])
    out = tmp_path / "build.zip"
    out.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        build_release.build(out)


# --------------------------------------------------------------------------- #
#  pinned-commit build (release TOCTOU fix): tracked_files_at + commit=       #
# --------------------------------------------------------------------------- #

def test_tracked_files_at_uses_ls_tree_with_the_given_commit(tmp_path, monkeypatch):
    """Git-free: tracked_files_at() must query the TREE at *commit* (git ls-tree), not
    the working tree / index (git ls-files) that tracked_files() reads."""
    calls = []

    class _Res:
        def __init__(self, out):
            self.stdout = out

    def fake_run(args, cwd=None, capture_output=None, check=None):
        calls.append(args)
        return _Res(b"VERSION\0localm/a.py\0")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)
    got = cm.tracked_files_at("deadbeef" * 5, repo=tmp_path)
    assert got == ["VERSION", "localm/a.py"]
    assert calls == [["git", "ls-tree", "-r", "-z", "--name-only", "deadbeef" * 5]]


def test_tracked_files_at_returns_none_when_commit_unresolvable(monkeypatch):
    """A commit that cannot be resolved (bad sha, gc'd object) must fail closed - the
    caller (build_release.build) turns this into a loud ValueError, never a silent
    fall-through to disk."""
    def fake_run(args, cwd=None, capture_output=None, check=None):
        raise subprocess.CalledProcessError(128, args)

    monkeypatch.setattr(cm.subprocess, "run", fake_run)
    assert cm.tracked_files_at("notacommit") is None


def test_build_commit_mode_uses_tracked_files_at_not_tracked_files(tmp_path, monkeypatch):
    """build(commit=...) must source the file LIST from tracked_files_at(commit) and
    the manifest from load_manifest_at(commit), never tracked_files()/load_manifest()
    (which would read the working tree's index/disk instead of the pinned commit)."""
    def boom_tracked(*a, **k):
        raise AssertionError("build(commit=...) must not call tracked_files() (disk-mode)")
    monkeypatch.setattr(cm, "tracked_files", boom_tracked)

    def boom_manifest(*a, **k):
        raise AssertionError("build(commit=...) must not call load_manifest() (disk-mode)")
    monkeypatch.setattr(cm, "load_manifest", boom_manifest)

    def boom_check(*a, **k):
        raise AssertionError("build(commit=...) must not call check_manifest() (disk-mode)")
    monkeypatch.setattr(cm, "check_manifest", boom_check)

    seen_commit = []
    monkeypatch.setattr(cm, "tracked_files_at", lambda commit, **k: (
        seen_commit.append(("tracked", commit)), ["VERSION", "pyproject.toml"])[1])
    monkeypatch.setattr(cm, "load_manifest_at", lambda commit, **k: (
        seen_commit.append(("manifest", commit)),
        {"include": ["VERSION", "pyproject.toml"], "exclude": [], "local_only": []})[1])

    def fake_archive(commit, repo):
        raise AssertionError("unused in this test")
    monkeypatch.setattr(build_release, "_git_archive_zip", fake_archive)

    # Reaching the missing-required-file / archive step proves tracked_files_at() and
    # load_manifest_at() were used; a stray call to the disk-mode equivalents would
    # raise via the boom_* patches first.
    with pytest.raises(AssertionError, match="unused in this test"):
        build_release.build(tmp_path / "build.zip", commit="cafef00d" * 5)
    assert seen_commit == [("tracked", "cafef00d" * 5), ("manifest", "cafef00d" * 5)]


def test_build_commit_mode_missing_required_file_still_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "tracked_files_at", lambda commit, **k: ["localm/a.py", "pyproject.toml"])
    monkeypatch.setattr(cm, "load_manifest_at", lambda commit, **k: {
        "include": ["localm/", "pyproject.toml"], "exclude": [], "local_only": []})
    with pytest.raises(ValueError, match="missing 'VERSION'"):
        build_release.build(tmp_path / "build.zip", commit="ab" * 20)


def test_build_commit_mode_uses_manifest_patterns_from_the_pinned_commit(tmp_path, monkeypatch):
    """build(commit=...) must classify members using load_manifest_at(commit)'s
    include/exclude patterns, not whatever release-manifest.toml looks like on disk -
    otherwise a manifest-pattern edit landing on disk after the commit was pinned
    could still change which of the pinned commit's tracked files get shipped."""
    monkeypatch.setattr(cm, "tracked_files_at", lambda commit, **k: [
        "VERSION", "pyproject.toml", "secret_dev_only.txt"])
    # The pinned commit's manifest does NOT include secret_dev_only.txt (it only
    # matches the exclude pattern - so it is cleanly excluded, not ambiguous)...
    monkeypatch.setattr(cm, "load_manifest_at", lambda commit, **k: {
        "include": ["VERSION", "pyproject.toml"],
        "exclude": ["secret_dev_only.txt"], "local_only": []})
    # ...while the LIVE-DISK manifest, if it were consulted, would ship it (proves the
    # live-disk manifest is never read in commit mode: this raises if called at all).
    def boom_manifest(*a, **k):
        raise AssertionError("build(commit=...) must not call load_manifest() (disk-mode)")
    monkeypatch.setattr(cm, "load_manifest", boom_manifest)

    # A minimal real in-memory archive standing in for git archive's output, so the
    # actual shipped membership can be inspected in the final zip rather than merely
    # inferred from a mock not firing.
    fake_bytes = io.BytesIO()
    with zipfile.ZipFile(fake_bytes, "w") as fz:
        fz.writestr("VERSION", "9.9.9\n")
        fz.writestr("pyproject.toml", "[project]\n")
        fz.writestr("secret_dev_only.txt", "should never ship\n")
    monkeypatch.setattr(build_release, "_git_archive_zip",
                        lambda commit, repo: zipfile.ZipFile(io.BytesIO(fake_bytes.getvalue())))

    out = tmp_path / "build.zip"
    monkeypatch.setattr(build_release, "_self_verify", lambda out: None)
    members = build_release.build(out, commit="beef" * 10)

    assert members == ["VERSION", "pyproject.toml"]
    shipped = set(zipfile.ZipFile(out).namelist())
    assert shipped == {"VERSION", "pyproject.toml"}
    assert "secret_dev_only.txt" not in shipped


def test_build_commit_mode_manifest_classification_problems_raise(tmp_path, monkeypatch):
    """An internally-inconsistent manifest AT THE PINNED COMMIT (e.g. a file matching
    both include and exclude) must block the build, exactly like the disk-mode gate."""
    monkeypatch.setattr(cm, "tracked_files_at", lambda commit, **k: ["shared/x.py"])
    monkeypatch.setattr(cm, "load_manifest_at", lambda commit, **k: {
        "include": ["shared/"], "exclude": ["shared/"], "local_only": []})
    with pytest.raises(ValueError, match="does not verify at the pinned commit"):
        build_release.build(tmp_path / "build.zip", commit="feed" * 10)


def test_tracked_files_at_unresolvable_commit_fails_loud_not_silent(tmp_path, monkeypatch):
    """build(commit=...) must raise (not silently fall back to disk) when the pinned
    commit cannot be resolved into a tree listing."""
    monkeypatch.setattr(cm, "tracked_files_at", lambda commit, **k: None)
    with pytest.raises(ValueError, match="could not list the git tree"):
        build_release.build(tmp_path / "build.zip", commit="0" * 40)


def test_load_manifest_at_unresolvable_commit_raises(tmp_path, monkeypatch):
    """build(commit=...) must raise (not silently fall back to disk) when the pinned
    commit's release-manifest.toml cannot be read - e.g. the commit predates the
    manifest, or is otherwise bad."""
    monkeypatch.setattr(cm, "tracked_files_at", lambda commit, **k: ["VERSION"])
    with pytest.raises(FileNotFoundError):
        build_release.build(tmp_path / "build.zip", commit="0" * 40)


def test_build_empty_string_commit_does_not_silently_fall_back_to_disk(tmp_path, monkeypatch):
    """NEGATIVE (the confirmed gap): build(commit="") must take the commit-mode path
    (and fail loud when "" cannot be resolved), not silently degrade to reading the
    live working tree the way a truthy/falsy ``if commit:`` check would. Proven by
    making the disk-mode-only functions explode if called at all."""
    def boom(*a, **k):
        raise AssertionError("build(commit=\"\") must not silently take the disk-mode path")
    monkeypatch.setattr(cm, "tracked_files", boom)
    monkeypatch.setattr(cm, "load_manifest", boom)
    monkeypatch.setattr(cm, "check_manifest", boom)

    with pytest.raises(ValueError, match="could not list the git tree"):
        build_release.build(tmp_path / "build.zip", commit="")


# --------------------------------------------------------------------------- #
#  the TOCTOU regression itself: pinned-commit bytes are immune to a          #
#  concurrent working-tree edit (finding: release build reads live disk)      #
# --------------------------------------------------------------------------- #

def _init_scratch_repo(root: Path) -> str:
    """A tiny throwaway git repo (NOT the real project checkout) with one committed
    file; returns its HEAD sha. Used to prove the pinned-commit read is immune to a
    disk edit without ever touching the real repo's tracked files."""
    import os
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    (root / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    subprocess.run(["git", "add", "VERSION"], cwd=root, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True, env=env).stdout.strip()


def test_git_archive_zip_reads_pinned_commit_not_a_later_disk_edit(tmp_path):
    """Core TOCTOU proof: pin a commit, THEN edit the same tracked file on disk
    (simulating an edit landing during the pre-publish CI-wait window) - the
    pinned-commit read must return the COMMITTED bytes, never the edited ones.

    Compared with line-ending normalized (``git archive`` legitimately applies the
    repo's own eol conversion, same as a checkout would - that is unrelated to the
    property under test here, which is disk-edit immunity, not byte-exact EOLs)."""
    sha = _init_scratch_repo(tmp_path)
    (tmp_path / "VERSION").write_text("9.9.9-TAMPERED\n", encoding="utf-8")

    archive = build_release._git_archive_zip(sha, tmp_path)
    content = archive.read("VERSION").decode("utf-8")
    assert content.strip() == "1.0.0"
    assert "TAMPERED" not in content


def test_git_archive_zip_unaffected_by_a_new_untracked_file(tmp_path):
    """A file dropped on disk after the commit was pinned (never committed) must not
    appear in the pinned-commit archive at all."""
    sha = _init_scratch_repo(tmp_path)
    (tmp_path / "SNEAKY.txt").write_text("should not ship\n", encoding="utf-8")

    archive = build_release._git_archive_zip(sha, tmp_path)
    assert "SNEAKY.txt" not in archive.namelist()


def test_load_manifest_at_reads_pinned_commit_not_a_later_disk_edit(tmp_path):
    """The manifest-pinning TOCTOU proof (the review-confirmed gap): reproduces the
    reviewers' exact exploit at the load_manifest_at() level, using real git against a
    throwaway scratch repo (not the real project checkout).

    Commit C's release-manifest.toml excludes tools/dev_secret.txt. Pin commit=C (as
    make_release.py does right after the clean-tree/HEAD gates). THEN, simulating an
    edit landing during the multi-minute CI wait, move tools/ from exclude to include
    in release-manifest.toml ON DISK (uncommitted). load_manifest_at(C) must still
    return C's ORIGINAL (excluding) patterns, not the tampered live-disk ones - proving
    the manifest content itself is pinned, not just the file list/bytes."""
    import os
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    manifest_c = (
        '[release.include]\npatterns = ["VERSION", "pyproject.toml"]\n\n'
        '[release.exclude]\npatterns = ["tools/"]\n\n'
        '[local_only]\npatterns = []\n'
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
    (tmp_path / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "release-manifest.toml").write_text(manifest_c, encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "dev_secret.txt").write_text("do not ship\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "C"], cwd=tmp_path, check=True, env=env)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                         text=True, check=True, env=env).stdout.strip()

    # Simulate a manifest edit landing on disk during the CI-wait window: move
    # tools/ from exclude to include (uncommitted).
    (tmp_path / "release-manifest.toml").write_text(
        manifest_c.replace(
            '[release.include]\npatterns = ["VERSION", "pyproject.toml"]',
            '[release.include]\npatterns = ["VERSION", "pyproject.toml", "tools/"]'
        ).replace('[release.exclude]\npatterns = ["tools/"]', '[release.exclude]\npatterns = []'),
        encoding="utf-8")

    pinned = cm.load_manifest_at(sha, repo=tmp_path)
    assert pinned["exclude"] == ["tools/"]
    assert "tools/" not in pinned["include"]

    # And the resulting membership (using the pinned tracked list + pinned manifest,
    # exactly as build() does in commit mode) must still exclude the secret file.
    tracked = cm.tracked_files_at(sha, repo=tmp_path)
    members = cm.release_files(tracked, pinned["include"], pinned["exclude"])
    assert "tools/dev_secret.txt" not in members
    assert set(members) == {"VERSION", "pyproject.toml"}


def test_build_commit_mode_rejects_a_tracked_symlink(tmp_path, monkeypatch):
    """A tracked git symlink (mode 120000) must not silently ship corrupted: git
    archive's "content" for a symlink entry is the raw target-PATH STRING, not the
    referenced file's bytes, so blindly copying it through (as the disk-mode branch's
    Path.is_file()-based dereferencing does NOT do here) would silently corrupt the
    file on a user's machine (plain zipfile.extractall() does not reconstruct
    symlinks). commit-mode must fail loud instead."""
    import os
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
    (tmp_path / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "real.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "VERSION", "pyproject.toml", "real.txt"],
                   cwd=tmp_path, check=True, env=env)
    # A real git-native symlink tree entry (mode 120000), added via plumbing so it
    # works even on a platform (Windows) without OS-level symlink support - this is
    # exactly how a POSIX contributor's `ln -s real.txt link.txt && git add link.txt`
    # would be stored.
    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], input=b"real.txt",
                          cwd=tmp_path, capture_output=True, check=True, env=env).stdout.decode().strip()
    subprocess.run(["git", "update-index", "--add", "--cacheinfo", f"120000,{blob},link.txt"],
                   cwd=tmp_path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "add a tracked symlink"],
                   cwd=tmp_path, check=True, env=env)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                         text=True, check=True, env=env).stdout.strip()

    monkeypatch.setattr(cm, "REPO", tmp_path)
    monkeypatch.setattr(cm, "tracked_files_at",
                        lambda commit, **k: ["VERSION", "pyproject.toml", "real.txt", "link.txt"])
    monkeypatch.setattr(cm, "load_manifest_at", lambda commit, **k: {
        "include": ["VERSION", "pyproject.toml", "real.txt", "link.txt"],
        "exclude": [], "local_only": []})

    with pytest.raises(ValueError, match="symlink"):
        build_release.build(tmp_path / "out.zip", commit=sha)


def test_build_commit_mode_rejects_a_tracked_submodule(tmp_path, monkeypatch):
    """A tracked git submodule (gitlink, mode 160000) must be explicitly rejected,
    not merely happen to fail via an unrelated check. git archive represents a
    gitlink as a DIRECTORY-PLACEHOLDER zip entry (name with a trailing slash), which
    used to only be caught as a side effect of the "member missing from archive"
    membership check (the bare tracked path never equals the trailing-slash archive
    name) - an accident of git's zip export format the code did not actually
    design for. Reject it explicitly via ZipInfo.is_dir() instead."""
    import os
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
    (tmp_path / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    subprocess.run(["git", "add", "VERSION", "pyproject.toml"], cwd=tmp_path, check=True, env=env)
    # A real git-native gitlink tree entry (mode 160000), added via plumbing so it
    # works without needing an actual .gitmodules-registered submodule - this is
    # exactly how a real `git submodule add` commits a gitlink entry.
    fake_sha = "a" * 40   # any 40-hex string is a valid gitlink target for this purpose
    subprocess.run(["git", "update-index", "--add", "--cacheinfo",
                    f"160000,{fake_sha},fakesubmodule"], cwd=tmp_path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "add a tracked gitlink"],
                   cwd=tmp_path, check=True, env=env)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                         text=True, check=True, env=env).stdout.strip()

    monkeypatch.setattr(cm, "REPO", tmp_path)
    monkeypatch.setattr(cm, "tracked_files_at",
                        lambda commit, **k: ["VERSION", "pyproject.toml", "fakesubmodule"])
    monkeypatch.setattr(cm, "load_manifest_at", lambda commit, **k: {
        "include": ["VERSION", "pyproject.toml", "fakesubmodule"],
        "exclude": [], "local_only": []})

    with pytest.raises(ValueError, match="directory|submodule"):
        build_release.build(tmp_path / "out.zip", commit=sha)


@pytest.mark.integration
def test_build_commit_mode_never_reads_bytes_from_disk(tmp_path, monkeypatch):
    """Structural proof against the REAL repo (read-only, no mutation): a
    pinned-commit build must never take the disk-reading branch (ZipFile.write on a
    real path) - only the archive-reading branch (ZipFile.writestr). Poisons the
    disk-read call so any accidental fallback to disk fails loudly instead of
    silently passing."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()

    def poisoned_write(self, filename, *a, **k):
        raise AssertionError(
            f"build(commit=...) read from disk (ZipFile.write) instead of the "
            f"pinned commit's git archive: {filename}")
    monkeypatch.setattr(zipfile.ZipFile, "write", poisoned_write)

    out = tmp_path / "pinned.zip"
    members = build_release.build(out, commit=head)   # must not raise despite the poison
    assert out.is_file()
    assert "VERSION" in members and "pyproject.toml" in members


@pytest.mark.integration
def test_pinned_commit_build_matches_disk_build_member_list(tmp_path):
    """Wiring sanity check on the real repo: a pinned-commit build and a live-disk
    build agree on WHICH files ship (membership depends on the tracked file list and
    the manifest patterns, not on working-tree content, so this holds regardless of
    any uncommitted edit in the dev tree)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    disk_members = build_release.build(tmp_path / "disk.zip")
    pinned_members = build_release.build(tmp_path / "pinned.zip", commit=head)
    assert disk_members == pinned_members


# --------------------------------------------------------------------------- #
#  Integration: real tree + real git + real build                            #
# --------------------------------------------------------------------------- #

@pytest.mark.integration
class TestRealTree:
    def test_real_tree_verifies_clean(self):
        """The whole point: the committed tree passes the gate with zero problems."""
        problems = cm.check_manifest()
        assert problems == [], "\n".join(problems)

    def test_every_tracked_file_classified_exactly_once(self):
        """Independently re-derive: every tracked file matches exactly one of
        include/exclude - no unclassified, no ambiguous."""
        man = cm.load_manifest()
        tracked = cm.tracked_files()
        assert tracked, "expected a git checkout with tracked files"
        for rel in tracked:
            inc = [p for p in man["include"] if cm._match(p, rel)]
            exc = [p for p in man["exclude"] if cm._match(p, rel)]
            assert len(inc) >= 1 or len(exc) >= 1, f"{rel} is unclassified"
            assert not (inc and exc), f"{rel} is in BOTH include ({inc}) and exclude ({exc})"

    def test_no_tracked_file_is_local_only(self):
        man = cm.load_manifest()
        tracked = cm.tracked_files()
        for rel in tracked:
            hit = [p for p in man["local_only"] if cm._match(p, rel)]
            assert not hit, f"{rel} is tracked but declared local-only via {hit}"

    def test_local_only_patterns_are_really_ignored(self):
        man = cm.load_manifest()
        assert cm._local_ignore_problems(man["local_only"]) == []

    def test_a_not_ignored_pattern_is_detected(self):
        """NEGATIVE: a local-only pattern git does NOT ignore must be flagged, so the
        cross-check cannot silently pass on drift."""
        problems = cm._local_ignore_problems(["definitely_not_ignored_xyz/"])
        assert any("not ignored by git" in p.lower() for p in problems), problems


@pytest.mark.integration
class TestRealBuild:
    def test_build_produces_valid_updater_artifact(self, tmp_path):
        out = tmp_path / "build.zip"
        members = build_release.build(out)
        assert out.is_file()
        # The updater's OWN acceptance gate must accept it.
        from localm import _apply_update as au
        au.verify_zip(out)
        names = set(zipfile.ZipFile(out).namelist())
        # required at root
        assert "VERSION" in names and "pyproject.toml" in names
        # product ships
        assert any(n.startswith("localm/") for n in names)
        assert any(n.startswith("docs/") for n in names)
        assert "scripts/report_issue.py" in names
        # dev-only never ships
        assert not any(n.startswith("tests/") for n in names)
        assert not any(n.startswith(".github/") for n in names)
        assert not any(n.startswith("tools/") for n in names)
        assert "release-manifest.toml" not in names
        assert "AGENTS.md" not in names
        assert set(members) == names

    def test_built_artifact_classifies_and_extracts(self, tmp_path):
        """End-to-end updater contract: the built zip extracts and classify() runs."""
        out = tmp_path / "build.zip"
        build_release.build(out)
        from localm import _apply_update as au
        from localm import updater
        root = au.extract(out, tmp_path / "staging")
        assert (root / "VERSION").read_text(encoding="utf-8").strip()
        klass = updater.classify(root, REPO_ROOT)
        assert klass in updater._ORDER   # a valid update class, no exception


# ---- HONESTY: the STANDALONE gate must FAIL when it classified nothing ----
# check_manifest() returns [] both when the repo is genuinely clean AND when git
# is unavailable (its documented library silence - build_release.py turns that
# None into its own hard error, so [] can never green-light a build). main() has
# no such caller: it PRINTS A VERDICT, so "Release manifest check passed." after
# scanning ZERO files is a false clean. Twin of test_check_hygiene.py's
# "HONESTY FIX #3" block, which covered the parent gate but not this entry point.

class TestStandaloneGateDoesNotPassWithoutScanning:
    def test_main_fails_loud_when_git_is_unavailable(self, monkeypatch, capsys):
        """NEGATIVE: git unusable -> main() must be non-zero and must NOT print
        'passed'. Before the fix this printed 'Release manifest check passed.'
        and returned 0 having classified nothing."""
        monkeypatch.setattr(cm, "tracked_files", lambda *a, **k: None)
        rc = cm.main([])
        out = capsys.readouterr()
        assert rc == 1
        assert "passed" not in out.out.lower()
        assert "nothing was scanned" in out.err.lower()

    def test_library_keeps_its_documented_silence(self, monkeypatch):
        """POSITIVE (do not over-escalate): the LIBRARY contract is unchanged -
        check_manifest() still returns [] when git is unavailable, because
        build_release.py relies on that and fails loud on its own."""
        monkeypatch.setattr(cm, "tracked_files", lambda *a, **k: None)
        assert cm.check_manifest() == []

    @pytest.mark.integration
    def test_main_still_passes_on_the_real_checkout(self):
        """The happy path must be untouched by the fail-loud guard."""
        assert cm.main([]) == 0
