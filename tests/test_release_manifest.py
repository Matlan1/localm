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
