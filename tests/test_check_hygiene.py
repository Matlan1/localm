# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py must scan .html for machine-absolute paths.

The absolute-path heuristic (check 3) is gated on _CODE_EXTS. These tests pin
that .html is in that set and that the shipped index.html is clean.
"""

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_html_drive_letter_is_detected(tmp_path, monkeypatch):
    """NEGATIVE: a drive-letter path in an .html must be flagged."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "page.html"
    p.write_text('<input placeholder="D:\\projects\\x">\n', encoding="utf-8")
    problems = ch._scan(p)
    assert any("absolute/machine path" in x for x in problems), problems


def test_html_neutral_placeholder_is_clean(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "page.html"
    p.write_text('<input placeholder="path to your project">\n', encoding="utf-8")
    assert not [x for x in ch._scan(p) if "absolute/machine path" in x]


def test_shipped_index_html_has_no_machine_paths():
    """The real GUI index.html must stay free of drive-letter placeholders."""
    ch = _load_check_hygiene()
    index = REPO_ROOT / "localm" / "plugins" / "gui" / "static" / "index.html"
    problems = [x for x in ch._scan(index) if "absolute/machine path" in x]
    assert not problems, problems


def test_html_is_in_code_exts():
    ch = _load_check_hygiene()
    assert ".html" in ch._CODE_EXTS


# ---- check 3: .js/.mjs/.yaml/.yml coverage ---------------------------------
# Check 3's absolute-path heuristic scans .js/.mjs/.yaml/.yml as well as .py.

def test_js_absolute_path_is_detected(tmp_path, monkeypatch):
    """NEGATIVE: a drive-letter path in a .js file must be flagged."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "app.js"
    p.write_text('const DEFAULT_DIR = "D:\\\\projects\\\\x";\n', encoding="utf-8")
    problems = ch._scan(p)
    assert any("absolute/machine path" in x for x in problems), problems


def test_mjs_absolute_path_is_detected(tmp_path, monkeypatch):
    """Same coverage for the .mjs extension (used by the frontend test/tooling
    scripts and playwright.config.mjs)."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "config.mjs"
    p.write_text('export const dir = "/home/someone/models";\n', encoding="utf-8")
    problems = ch._scan(p)
    assert any("absolute/machine path" in x for x in problems), problems


def test_yaml_absolute_path_is_detected(tmp_path, monkeypatch):
    """Same coverage for .yaml/.yml (CI workflows, dependabot config, etc.)."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "workflow.yml"
    p.write_text('path: "Z:\\\\Users\\\\someone\\\\build"\n', encoding="utf-8")
    problems = ch._scan(p)
    assert any("absolute/machine path" in x for x in problems), problems


def test_js_neutral_placeholder_is_clean(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "app.js"
    p.write_text('const label = "path to your project";\n', encoding="utf-8")
    assert not [x for x in ch._scan(p) if "absolute/machine path" in x]


def test_js_and_yaml_are_in_code_exts():
    ch = _load_check_hygiene()
    for ext in (".js", ".mjs", ".yaml", ".yml"):
        assert ext in ch._CODE_EXTS


def test_frontend_test_suite_fixtures_stay_exempt(tmp_path, monkeypatch):
    """The frontend suites' own conventions (tests-js/*.test.mjs,
    tests-e2e/*.spec.mjs), which the Python is_test heuristic does not match,
    are exempt from the absolute-path heuristic exactly as tests/test_*.py
    is."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    for rel in ("tests-js/upload.test.mjs", "tests-e2e/boot-and-click.spec.mjs"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('const dir = "/home/uploads";\n', encoding="utf-8")
        problems = ch._scan(p)
        assert not [x for x in problems if "absolute/machine path" in x], (rel, problems)


def test_shipped_frontend_js_has_no_machine_paths():
    """The real GUI/CLI JS files must stay free of drive-letter or /home//Users
    absolute paths."""
    ch = _load_check_hygiene()
    tracked = ch._tracked_files()
    problems = []
    for f in tracked:
        if f.suffix.lower() in (".js", ".mjs"):
            problems.extend(x for x in ch._scan(f) if "absolute/machine path" in x)
    assert not problems, problems


# ---- check 2: secret disclosure --------------------------------------------
# The synthetic tokens below are assembled from fragments at runtime so this
# file does not itself contain a literal secret for the hygiene check to flag.

def test_finegrained_github_pat_is_detected(tmp_path, monkeypatch):
    """NEGATIVE: a modern fine-grained GitHub PAT (github_pat_11...) must be
    flagged as a disclosure."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    token = "github" + "_pat_" + "11" + "A" * 60
    p = tmp_path / "leak.txt"
    p.write_text(f"WRANGLER_SECRET={token}\n", encoding="utf-8")
    problems = ch._scan(p)
    assert any("disclosure" in x for x in problems), problems


def test_classic_github_pat_still_detected(tmp_path, monkeypatch):
    """Regression: the classic ghp_ token must keep being flagged."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    token = "ghp" + "_" + "B" * 36
    p = tmp_path / "leak.txt"
    p.write_text(f"GH_TOKEN={token}\n", encoding="utf-8")
    problems = ch._scan(p)
    assert any("disclosure" in x for x in problems), problems


def test_github_pat_mention_without_token_is_clean(tmp_path, monkeypatch):
    """The bare 'github_pat_...' placeholder used in docs (no real token body)
    must NOT be flagged: the pattern requires 20+ token chars after the
    prefix, so a mention like 'starts with github_pat_...' stays clean."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "doc.md"
    p.write_text("Copy the token (starts with `github" + "_pat_...`).\n",
                 encoding="utf-8")
    problems = ch._scan(p)
    assert not [x for x in problems if "disclosure" in x], problems


# ---- CHANGELOG append-only guard -------------------------------------------
# The guard diffs the working CHANGELOG against the last committed version and
# fails if a shipped entry line disappeared. Headers and link-reference
# definitions are exempt.

import subprocess

_BASE_CHANGELOG = (
    "# Changelog\n\n"
    "This file is append-only.\n\n"
    "## [Unreleased]\n\n"
    "### Added\n"
    "- work in progress\n\n"
    "## [0.1.0] - 2026-07-04\n\n"
    "First tagged release.\n\n"
    "### Added\n"
    "- inference and CLI\n"
    "- the GUI\n\n"
    "[Unreleased]: https://example.invalid/compare/v0.1.0...HEAD\n"
    "[0.1.0]: https://example.invalid/releases/tag/v0.1.0\n"
)


def test_changelog_removed_lines_no_change_is_clean():
    """No change to the changelog -> nothing removed."""
    ch = _load_check_hygiene()
    assert ch._changelog_removed_lines(_BASE_CHANGELOG, _BASE_CHANGELOG) == []


def test_changelog_removed_lines_add_section_on_top_is_clean():
    """Adding a brand-new version section on top (the normal release) removes
    nothing: every prior entry line is still present. MUST NOT false-positive."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace(
        "## [Unreleased]\n\n### Added\n- work in progress\n",
        "## [Unreleased]\n\n## [0.2.0] - 2026-08-01\n\n### Added\n"
        "- work in progress\n- another shipped thing\n")
    assert ch._changelog_removed_lines(_BASE_CHANGELOG, new) == []


def test_changelog_removed_lines_deleting_an_entry_fails():
    """NEGATIVE: deleting an existing shipped entry line is flagged."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace("- the GUI\n", "")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["- the GUI"], removed


# The gate uses "a tag exists" as its proxy for "it shipped", so a version that
# is tagged but not yet published gets an explicit carve-out.
_DRAFT_AND_SHIPPED_CHANGELOG = (
    "# Changelog\n\n"
    "## [Unreleased]\n\n"
    "- work in progress\n\n"
    "## [0.1.5rc1] - 2026-08-07\n\n"
    "### Added\n"
    "- a thing that was never published\n\n"
    "## [0.1.0] - 2026-07-04\n\n"
    "### Added\n"
    "- a thing that really shipped\n"
)


def test_a_never_published_version_section_is_not_frozen():
    """0.1.5rc1 was tagged but its GitHub release stayed a DRAFT, so nobody could have
    downloaded it and there is no public history to protect. Removing its section is a
    correction, not a history rewrite."""
    ch = _load_check_hygiene()
    new = _DRAFT_AND_SHIPPED_CHANGELOG.replace(
        "## [0.1.5rc1] - 2026-08-07\n\n### Added\n"
        "- a thing that was never published\n\n", "")
    assert ch._changelog_removed_lines(_DRAFT_AND_SHIPPED_CHANGELOG, new) == []


def test_the_never_published_carve_out_does_not_unfreeze_a_real_release():
    """Deleting from a genuinely published section is still flagged, in the same
    file where the draft section is being removed."""
    ch = _load_check_hygiene()
    new = _DRAFT_AND_SHIPPED_CHANGELOG.replace("- a thing that really shipped\n", "")
    removed = ch._changelog_removed_lines(_DRAFT_AND_SHIPPED_CHANGELOG, new)
    assert removed == ["- a thing that really shipped"], removed


def test_the_never_published_list_holds_only_provably_unpublished_versions():
    """Every entry in _NEVER_PUBLISHED_VERSIONS is a release that was never
    promoted out of draft, and no shipped version appears in it."""
    ch = _load_check_hygiene()
    assert ch._NEVER_PUBLISHED_VERSIONS == frozenset({"0.1.5rc1"})
    for shipped in ("0.1.0", "0.1.4", "0.1.5rc2", "0.1.5rc3"):
        assert shipped not in ch._NEVER_PUBLISHED_VERSIONS


def test_changelog_rewriting_unreleased_is_allowed():
    """The in-progress [Unreleased] draft is freely rewritable until it is cut into a
    version: rewording or dropping an [Unreleased] line is NOT a history rewrite."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace(
        "### Added\n- work in progress\n\n## [0.1.0]",
        "### Changed\n- reworded and expanded the in-progress notes\n\n## [0.1.0]")
    assert ch._changelog_removed_lines(_BASE_CHANGELOG, new) == []


def test_changelog_rewriting_a_published_entry_fails():
    """A PUBLISHED (versioned) entry is frozen: rewriting its wording is a history
    rewrite and is flagged, even though nothing is deleted outright."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace("- the GUI\n", "- the GUI (reworded)\n")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["- the GUI"], removed


def test_changelog_redating_a_published_version_header_fails():
    """NEGATIVE: a published version header carries shipped-record content
    (version + date) like any body entry, so re-dating one in place is
    flagged."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace(
        "## [0.1.0] - 2026-07-04", "## [0.1.0] - 2026-07-09")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["## [0.1.0] - 2026-07-04"], removed


def test_changelog_renumbering_a_published_version_header_fails():
    """The other direction: renumbering a shipped version (e.g. passing off
    0.1.0 as 0.1.1) is caught too."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace(
        "## [0.1.0] - 2026-07-04", "## [0.1.1] - 2026-07-04")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["## [0.1.0] - 2026-07-04"], removed


def test_git_helper_decodes_output_as_utf8_not_locale_default(monkeypatch):
    """_git() decodes git's output as UTF-8, not the platform locale default.

    The changelog append-only guard reads the baseline via `git show
    <ref>:CHANGELOG.md` (through _git) and the working tree via
    read_text(encoding="utf-8"), so the two reads must use the same encoding.
    Asserted on the kwarg rather than behaviourally, since a behavioural test
    would only bite on a cp1252 box."""
    ch = _load_check_hygiene()
    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(argv, **kwargs):
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(ch.subprocess, "run", _fake_run)
    ch._git("show", "HEAD:CHANGELOG.md")
    assert captured.get("encoding") == "utf-8", captured


def test_changelog_renaming_a_published_subsection_header_fails():
    """NEGATIVE: a '### Added' subsection header inside an already-published
    section is in the protected set, so flipping it to '### Removed' - which
    misrepresents what that release shipped while leaving the bullet text
    unchanged - is flagged."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace("### Added\n- inference and CLI\n- the GUI",
                                  "### Removed\n- inference and CLI\n- the GUI")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["### Added"], removed


def test_changelog_deleting_a_published_subsection_header_fails():
    """The other direction: deleting '### Added' entirely, orphaning its bullets
    directly under the version header, is caught too."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace("### Added\n- inference and CLI\n- the GUI",
                                  "- inference and CLI\n- the GUI")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["### Added"], removed


def test_changelog_unreleased_subsection_headers_stay_freely_editable():
    """A subsection header under '## [Unreleased]' (not yet published) must NOT
    become protected - only ones inside an already-published version section."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace(
        "## [Unreleased]\n\n### Added\n- work in progress\n",
        "## [Unreleased]\n\n### Changed\n- work in progress\n")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == [], removed


def test_changelog_unreleased_header_stays_freely_editable():
    """The '## [Unreleased]' header must NOT become protected - only VERSIONED
    headers (a leading digit inside the brackets) are frozen."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace("## [Unreleased]", "## [Unreleased draft]")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == [], removed


def test_changelog_removed_lines_release_rename_is_clean():
    """Cutting a release renames the `## [Unreleased]` header, adds a version
    header, and rewrites the compare link + adds a new tag link. Only headers and
    link-reference lines change; no content entry is removed -> clean."""
    ch = _load_check_hygiene()
    new = (_BASE_CHANGELOG
           .replace("## [Unreleased]\n\n### Added\n- work in progress\n",
                    "## [Unreleased]\n\n## [0.2.0] - 2026-08-01\n\n### Added\n"
                    "- work in progress\n")
           .replace("[Unreleased]: https://example.invalid/compare/v0.1.0...HEAD\n",
                    "[Unreleased]: https://example.invalid/compare/v0.2.0...HEAD\n"
                    "[0.2.0]: https://example.invalid/releases/tag/v0.2.0\n"))
    assert ch._changelog_removed_lines(_BASE_CHANGELOG, new) == []


def _init_changelog_repo(tmp_path, text):
    """A throwaway git repo with CHANGELOG.md committed, so the git-wired guard has
    a real HEAD baseline to diff the working tree against."""
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    import os
    run_env = {**os.environ, **env}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=run_env)
    (tmp_path / "CHANGELOG.md").write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "CHANGELOG.md"], cwd=tmp_path, check=True, env=run_env)
    subprocess.run(["git", "commit", "-qm", "add changelog"], cwd=tmp_path,
                   check=True, env=run_env)


def test_changelog_append_only_guard_flags_a_committed_baseline_deletion(tmp_path, monkeypatch):
    """The git-wired entrypoint: with CHANGELOG committed, deleting a shipped entry
    line in the working tree is flagged; no-change and add-on-top pass."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_changelog_repo(tmp_path, _BASE_CHANGELOG)
    cl = tmp_path / "CHANGELOG.md"

    # no change -> clean
    assert ch._changelog_append_only() == []

    # add a section on top -> clean (the normal release; must not false-positive)
    cl.write_text(_BASE_CHANGELOG.replace(
        "## [Unreleased]\n",
        "## [Unreleased]\n\n## [0.2.0] - 2026-08-01\n\n### Added\n- shipped\n"),
        encoding="utf-8")
    assert ch._changelog_append_only() == []

    # delete an existing entry -> FAIL
    cl.write_text(_BASE_CHANGELOG.replace("- the GUI\n", ""), encoding="utf-8")
    problems = ch._changelog_append_only()
    assert problems and any("append-only" in p for p in problems), problems


def test_changelog_append_only_guard_flags_a_published_header_rewrite(tmp_path, monkeypatch):
    """The git-wired entrypoint (the actual CI / pre-commit-hook gate) catches a
    header-only rewrite of a published section, not just the pure
    _changelog_removed_lines layer."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_changelog_repo(tmp_path, _BASE_CHANGELOG)
    cl = tmp_path / "CHANGELOG.md"

    cl.write_text(
        _BASE_CHANGELOG.replace("## [0.1.0] - 2026-07-04", "## [0.1.0] - 2026-07-09"),
        encoding="utf-8")
    problems = ch._changelog_append_only()
    assert problems and any("append-only" in p for p in problems), problems


def test_changelog_append_only_guard_catches_a_committed_deletion_in_ci(tmp_path, monkeypatch):
    """CI / clean-checkout path: even after the deletion is COMMITTED (working ==
    HEAD, so a plain working-vs-HEAD diff would see nothing), a shipped entry
    dropped relative to origin/master is still caught via the merge-base baseline.
    An add-on-top commit on the same clean tree must still pass (no false-positive
    on a new release landing on the branch)."""
    import os
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, env=env,
                              capture_output=True, text=True)

    git("init", "-q")
    cl = tmp_path / "CHANGELOG.md"
    cl.write_text(_BASE_CHANGELOG, encoding="utf-8")
    git("add", "CHANGELOG.md")
    git("commit", "-qm", "published record")
    base_sha = git("rev-parse", "HEAD").stdout.strip()
    # simulate the published master WITHOUT needing a real remote
    git("update-ref", "refs/remotes/origin/master", base_sha)

    # add a new section on top and COMMIT it: clean tree, but append-only -> clean
    cl.write_text(_BASE_CHANGELOG.replace(
        "## [Unreleased]\n",
        "## [Unreleased]\n\n## [0.2.0] - 2026-08-01\n\n### Added\n- shipped\n"),
        encoding="utf-8")
    git("add", "CHANGELOG.md")
    git("commit", "-qm", "cut 0.2.0")
    assert ch._changelog_append_only() == [], "add-on-top commit must pass"

    # now DELETE a shipped entry and COMMIT it: working == HEAD, but it is gone
    # relative to the origin/master baseline -> must be flagged
    cl.write_text((tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
                  .replace("- the GUI\n", ""), encoding="utf-8")
    git("add", "CHANGELOG.md")
    git("commit", "-qm", "oops deleted a line")
    problems = ch._changelog_append_only()
    assert problems and any("append-only" in p for p in problems), problems


def test_changelog_append_only_guard_passes_without_a_git_baseline(tmp_path, monkeypatch):
    """A brand-new (never-committed) CHANGELOG has no baseline to diff against, so
    the guard passes rather than crashing or blocking the first commit."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "CHANGELOG.md").write_text(_BASE_CHANGELOG, encoding="utf-8")
    assert ch._changelog_append_only() == []


def test_changelog_append_only_no_false_positive_on_non_ascii_under_cp1252(
        tmp_path, monkeypatch):
    """A shipped entry with NON-ASCII content (an emoji, an accented word) must
    not be reported as removed just because the process locale is not UTF-8.

    ``_git`` compares ``git show <ref>:CHANGELOG.md`` against the working file
    read as UTF-8, and decodes explicitly as UTF-8 rather than with the locale
    codepage. Forcing a cp1252 locale exercises that on any platform."""
    import locale
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    # A shipped entry line carrying both an emoji and an accented word.
    non_ascii = _BASE_CHANGELOG.replace(
        "- the GUI\n",
        "- the GUI, now with a \U0001f9e0 memory chip and an äccented note\n")
    _init_changelog_repo(tmp_path, non_ascii)
    # working tree is byte-identical to HEAD -> nothing removed, on any locale.
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    assert ch._changelog_append_only() == []
    # The raw git read round-trips the non-ASCII intact.
    got = ch._git("show", "HEAD:CHANGELOG.md")
    assert got is not None and got.returncode == 0
    assert "\U0001f9e0" in got.stdout and "äccented" in got.stdout


def test_real_changelog_is_append_only_against_head():
    """The real repo CHANGELOG must itself satisfy the guard."""
    ch = _load_check_hygiene()
    assert ch._changelog_append_only() == []


# ---- the pending (cut but UNRELEASED) section is a draft, not history --------
# A section whose version has no tag yet is a draft. These pin both directions:
# the pending section is editable, and everything else stays frozen even when
# tags are absent.

def _init_versioned_repo(tmp_path, version, changelog=_BASE_CHANGELOG):
    """A throwaway repo with a committed CHANGELOG and a VERSION file naming *version*."""
    _init_changelog_repo(tmp_path, changelog)
    (tmp_path / "VERSION").write_text(f"{version}\n", encoding="utf-8")


def _tag(tmp_path, name):
    import os
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "tag", name], cwd=tmp_path, check=True, env=env)


def test_pending_unreleased_section_may_be_recut(tmp_path, monkeypatch):
    """VERSION names 0.1.0 and no v0.1.0 tag exists -> that section is the pending
    release, still a draft: re-dating it and rewriting its entries is allowed."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_versioned_repo(tmp_path, "0.1.0")
    assert ch._pending_release_version() == "0.1.0"

    (tmp_path / "CHANGELOG.md").write_text(
        _BASE_CHANGELOG.replace("## [0.1.0] - 2026-07-04", "## [0.1.0] - 2026-07-17")
                       .replace("- the GUI\n", "- the GUI, reworded\n"),
        encoding="utf-8")
    assert ch._changelog_append_only() == [], "an unreleased draft must be re-cuttable"


def test_pending_section_refreezes_once_tagged(tmp_path, monkeypatch):
    """The SAME edit is a history rewrite once v0.1.0 exists: the tag is what makes a
    section real history, so the guard must bite again the moment it appears."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_versioned_repo(tmp_path, "0.1.0")
    _tag(tmp_path, "v0.1.0")
    assert ch._pending_release_version() is None

    (tmp_path / "CHANGELOG.md").write_text(
        _BASE_CHANGELOG.replace("## [0.1.0] - 2026-07-04", "## [0.1.0] - 2026-07-17"),
        encoding="utf-8")
    problems = ch._changelog_append_only()
    assert problems and any("append-only" in p for p in problems), problems


def test_older_section_stays_frozen_even_with_no_tags(tmp_path, monkeypatch):
    """The safety property that makes the tag lookup safe: a section whose version is
    NOT the current VERSION is frozen unconditionally, so a clone with no tags at all
    (a --no-tags / shallow clone) can still never rewrite real history."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_versioned_repo(tmp_path, "0.2.0")     # 0.1.0 is NOT the pending release
    assert ch._pending_release_version() == "0.2.0"

    (tmp_path / "CHANGELOG.md").write_text(
        _BASE_CHANGELOG.replace("- the GUI\n", ""), encoding="utf-8")
    problems = ch._changelog_append_only()
    assert problems and any("append-only" in p for p in problems), problems


def test_missing_version_file_freezes_everything(tmp_path, monkeypatch):
    """Fail SAFE: with no VERSION file the guard cannot identify a pending release, so
    it protects every version section rather than guessing."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_changelog_repo(tmp_path, _BASE_CHANGELOG)     # no VERSION file written
    assert ch._pending_release_version() is None

    (tmp_path / "CHANGELOG.md").write_text(
        _BASE_CHANGELOG.replace("- the GUI\n", ""), encoding="utf-8")
    problems = ch._changelog_append_only()
    assert problems and any("append-only" in p for p in problems), problems


# ---- check 5: raw single-resource accessor guard ----------------------------
# _raw_accessor_violations flags a call site re-adopting the raw single-resource
# accessor of a "single -> combined N resources" capability (vram_info() ->
# vram_capacity()). These pin both direct-call and import-alias detection.

_GUARD_NAME = "vram_info"
_GUARD_SPEC_KEY = "wrapper"


def _guarded_wrapper_text(ch):
    return ch._RAW_ACCESSOR_GUARDS[_GUARD_NAME][_GUARD_SPEC_KEY]


def test_direct_call_outside_allowlist_is_flagged(tmp_path, monkeypatch):
    """NEGATIVE: a plain `from localm.discover import vram_info` + a direct
    call, in a file NOT in the guard's allowed set, must be flagged."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "some_module.py"
    p.write_text(
        "from localm.discover import vram_info\n"
        "def f():\n"
        "    return vram_info().get('total')\n",
        encoding="utf-8")
    problems = ch._raw_accessor_violations([p])
    assert len(problems) == 1
    assert "some_module.py:3" in problems[0]
    assert _guarded_wrapper_text(ch) in problems[0]


def test_import_alias_bypass_is_flagged(tmp_path, monkeypatch):
    """NEGATIVE: `from localm.discover import vram_info as vi` followed by
    `vi()` must ALSO be flagged, since an ast.ImportFrom alias rebinds the name
    locally."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "some_module.py"
    p.write_text(
        "from localm.discover import vram_info as _vi\n"
        "def f():\n"
        "    return _vi().get('total')\n",
        encoding="utf-8")
    problems = ch._raw_accessor_violations([p])
    assert len(problems) == 1
    assert "some_module.py:3" in problems[0]
    assert "'_vi'" in problems[0]


def test_module_attribute_call_is_flagged_regardless_of_module_alias(tmp_path, monkeypatch):
    """`import localm.discover as disc; disc.vram_info()` must be flagged too:
    ast.Attribute.attr is the literal accessor name no matter what the MODULE
    is aliased to."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "some_module.py"
    p.write_text(
        "import localm.discover as disc\n"
        "def f():\n"
        "    return disc.vram_info().get('total')\n",
        encoding="utf-8")
    problems = ch._raw_accessor_violations([p])
    assert len(problems) == 1
    assert "some_module.py:3" in problems[0]


def test_call_inside_an_allowed_file_is_not_flagged(tmp_path, monkeypatch):
    """A file listed in the guard's allowed set (a documented single-resource
    exception) must NOT be flagged, direct call or alias alike."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    allowed_rel = next(iter(ch._RAW_ACCESSOR_GUARDS[_GUARD_NAME]["allowed"]))
    p = tmp_path / allowed_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "from localm.discover import vram_info\n"
        "def f():\n"
        "    return vram_info().get('free')\n",
        encoding="utf-8")
    assert ch._raw_accessor_violations([p]) == []


def test_call_inside_tests_directory_is_not_flagged(tmp_path, monkeypatch):
    """tests/ is exempt everywhere: a test legitimately calls/mocks the raw
    accessor directly to test IT, not just its consumers."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "tests" / "test_something.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "from localm.discover import vram_info\n"
        "def test_x():\n"
        "    assert vram_info() is not None\n",
        encoding="utf-8")
    assert ch._raw_accessor_violations([p]) == []


def test_docstring_or_comment_mention_is_not_flagged(tmp_path, monkeypatch):
    """A prose mention of 'vram_info()' in a docstring/comment (not an actual
    call) must NOT be flagged - this is an AST-based check, not text matching,
    specifically so it never false-positives on documentation."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "some_module.py"
    p.write_text(
        '"""See vram_info() for the single-GPU number this wraps."""\n'
        "# vram_info() is single-GPU only, per its own docstring.\n"
        "def f():\n"
        "    return 1\n",
        encoding="utf-8")
    assert ch._raw_accessor_violations([p]) == []


def test_real_tree_has_no_raw_accessor_violations():
    """Regression guard: the real, shipped tree must itself satisfy this check
    (every consumer either uses vram_capacity() or is in the allowed set)."""
    ch = _load_check_hygiene()
    tracked = ch._tracked_files()
    assert ch._raw_accessor_violations(tracked) == []


# ---- an unparseable .py is reported, not silently skipped -------------------
# The raw-accessor guard appends a problem for any .py it cannot read or parse
# (OSError / SyntaxError / UnicodeDecodeError) instead of skipping it.

def test_unparseable_py_is_flagged_not_silently_skipped(tmp_path, monkeypatch):
    """NEGATIVE: a .py with a syntax error the guard cannot ast.parse must be reported
    as unchecked, not skipped - a guard that 'passes' a file it never scanned is the
    exact silent pass rule 5 forbids."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "broken.py"
    p.write_text("def broken(:\n    pass\n", encoding="utf-8")   # deliberate syntax error
    problems = ch._raw_accessor_violations([p])
    assert problems, "an unparseable .py must produce a problem, not an empty (clean) result"
    assert any("could not read/parse" in x for x in problems), problems
    assert any("broken.py" in x for x in problems), problems


def test_valid_py_with_no_raw_call_still_clean(tmp_path, monkeypatch):
    """A parseable .py with no raw-accessor call stays clean: the report path
    fires ONLY on a genuinely unparseable file."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "fine.py"
    p.write_text("def f():\n    return 1 + 2\n", encoding="utf-8")
    assert ch._raw_accessor_violations([p]) == []


# ---- the top-level gate fails when it scanned nothing -----------------------
# When `git ls-files` fails (not a checkout, git missing), _tracked_files()
# returns [] and every scan runs over zero files; main() must not report a pass.

def test_hygiene_main_fails_when_nothing_scanned(monkeypatch):
    """NEGATIVE: with no tracked files enumerable (simulated git failure), main() must
    return non-zero and NOT report success."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "_tracked_files", lambda: [])
    assert ch.main([]) == 1


def test_hygiene_main_passes_when_files_are_scanned(tmp_path, monkeypatch):
    """With a real (clean) file to scan and the other sub-gates stubbed clean,
    main() returns 0: the scanned-nothing guard fires only when nothing was
    scanned."""
    ch = _load_check_hygiene()
    clean = tmp_path / "ok.py"
    clean.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_tracked_files", lambda: [clean])
    monkeypatch.setattr(ch, "_changelog_append_only", lambda: [])
    monkeypatch.setattr(ch, "_raw_accessor_violations", lambda t: [])
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])
    assert ch.main([]) == 0


# ---- release-manifest gate wiring -------------------------------------------
# These pin that main() actually calls the release-manifest gate
# (scripts/check_manifest.py).

def test_release_manifest_gate_failures_fail_the_build(monkeypatch):
    """A real classification problem from _release_manifest_gate() surfaces as a
    hygiene FAILURE (exit 1) rather than being dropped."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "_tracked_files", lambda: [Path("ok.py")])
    monkeypatch.setattr(ch, "_scan", lambda f: [])
    monkeypatch.setattr(ch, "_changelog_append_only", lambda: [])
    monkeypatch.setattr(ch, "_raw_accessor_violations", lambda t: [])
    monkeypatch.setattr(ch, "_big_test_write_violations", lambda t: [])
    monkeypatch.setattr(ch, "_sw_cache_derivation_violations", lambda: [])
    monkeypatch.setattr(ch, "_import_cycle_violations", lambda: [])
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])
    monkeypatch.setattr(ch, "_release_manifest_gate",
                        lambda: (["release.exclude pattern 'X' matches no tracked file"], []))
    assert ch.main([]) == 1


def test_release_manifest_gate_warns_but_passes_by_default_and_strict_escalates(monkeypatch, capsys):
    """A WARNING from _release_manifest_gate() (the expected shape when
    check_manifest.py is absent, e.g. on CI/most external clones) does not fail
    the default run, but DOES escalate under --strict - same contract as the
    other warn-only checks (4b)."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "_tracked_files", lambda: [Path("ok.py")])
    monkeypatch.setattr(ch, "_scan", lambda f: [])
    monkeypatch.setattr(ch, "_changelog_append_only", lambda: [])
    monkeypatch.setattr(ch, "_raw_accessor_violations", lambda t: [])
    monkeypatch.setattr(ch, "_big_test_write_violations", lambda t: [])
    monkeypatch.setattr(ch, "_sw_cache_derivation_violations", lambda: [])
    monkeypatch.setattr(ch, "_import_cycle_violations", lambda: [])
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])
    monkeypatch.delenv("LOCALM_HYGIENE_STRICT", raising=False)
    # Stubs the gate to WARN and asserts the warn-vs-strict escalation. A neutral
    # ([], []) stub above this line would be dead, overridden here.
    monkeypatch.setattr(ch, "_release_manifest_gate",
                        lambda: ([], ["release-manifest gate SKIPPED: ..."]))

    assert ch.main([]) == 0
    assert "release-manifest gate SKIPPED" in capsys.readouterr().err

    assert ch.main(["--strict"]) == 1
    assert "release-manifest gate SKIPPED" in capsys.readouterr().err


def test_release_manifest_gate_reports_a_warning_when_check_manifest_not_importable(monkeypatch):
    """Real (non-stubbed) behavior of _release_manifest_gate() itself: when
    scripts/check_manifest.py cannot be imported - the normal state on CI and
    most external clones, where it is gitignored - it returns a warning rather
    than (([], [])) as if there were nothing to report."""
    ch = _load_check_hygiene()
    monkeypatch.setitem(ch.sys.modules, "check_manifest", None)   # forces ImportError
    failures, warnings = ch._release_manifest_gate()
    assert failures == []
    assert len(warnings) == 1
    assert "check_manifest.py" in warnings[0]
    assert "SKIPPED" in warnings[0]
    # A None entry in sys.modules raises ModuleNotFoundError, which is the ABSENT
    # state, so this belongs on the SKIPPED branch.


def test_release_manifest_gate_does_not_claim_absent_when_the_checker_is_broken(
        monkeypatch):
    """A PRESENT but unimportable checker must not be reported as absent: absent
    is the expected gitignored case, while present-and-broken means the gate
    CANNOT RUN on the one machine that actually has the checker.

    A plain ImportError (not ModuleNotFoundError) is the realistic shape here: a
    broken internal import inside an otherwise-present check_manifest.py.
    """
    import importlib.abc

    ch = _load_check_hygiene()

    class _BrokenOnImport(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "check_manifest":
                raise ImportError("cannot import name 'nope' from 'os'")
            return None

    monkeypatch.delitem(ch.sys.modules, "check_manifest", raising=False)
    monkeypatch.setattr(ch.sys, "meta_path", [_BrokenOnImport(), *ch.sys.meta_path])

    failures, warnings = ch._release_manifest_gate()

    assert failures == []
    assert len(warnings) == 1
    assert "COULD NOT RUN" in warnings[0], warnings[0]
    assert "failed to import" in warnings[0], warnings[0]
    # It must not repeat the absent-file claim about a file that is present.
    assert "is not present in this checkout" not in warnings[0], warnings[0]


def test_release_manifest_gate_runs_the_real_checker_when_importable():
    """Real (non-stubbed) behavior when scripts/check_manifest.py IS importable
    (true in this dev checkout): _release_manifest_gate() actually calls
    check_manifest.check_manifest()."""
    ch = _load_check_hygiene()
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in ch.sys.path:
        ch.sys.path.insert(0, scripts_dir)
    # scripts/check_manifest.py is gitignored, so it is absent from CI and from
    # external clones; skip rather than fail when it is not importable.
    cm = pytest.importorskip(
        "check_manifest",
        reason="scripts/check_manifest.py is gitignored and absent from this checkout")
    failures, warnings = ch._release_manifest_gate()
    assert warnings == []
    # _release_manifest_gate() must report the same as check_manifest
    # .check_manifest() does against the real tree, proving it is a passthrough.
    assert failures == list(cm.check_manifest())


# ---- [Unreleased] draft corruption WARNINGS ---------------------------------
# The append-only gate exempts the [Unreleased] draft, so two warn-only
# detectors cover it: a DROP (a baseline draft line gone from the working copy)
# and a new DUPLICATE. Both warn with exit 0 and fail only under --strict /
# LOCALM_HYGIENE_STRICT=1.

_DRAFT_BASE = (
    "# Changelog\n\n"
    "intro text outside any section\n\n"
    "## [Unreleased]\n\n"
    "### Added\n"
    "- my own draft bullet\n"
    "- sibling bullet a rebase must not eat\n"
    "- a wrapped draft bullet whose second line\n"
    "  carries the rest of the sentence\n\n"
    "## [0.1.0] - 2026-07-04\n\n"
    "### Added\n"
    "- inference and CLI\n\n"
    "[Unreleased]: https://example.invalid/compare/v0.1.0...HEAD\n"
    "[0.1.0]: https://example.invalid/releases/tag/v0.1.0\n"
)


def test_unreleased_lines_collects_draft_content_only():
    """The draft extractor takes bullets AND their wrapped continuation lines
    from [Unreleased], and nothing else: no headers (## / ###), no blank lines,
    no link-reference definitions, no intro text, no published-section lines."""
    ch = _load_check_hygiene()
    got = ch._changelog_unreleased_lines(_DRAFT_BASE)
    assert got == [
        "- my own draft bullet",
        "- sibling bullet a rebase must not eat",
        "- a wrapped draft bullet whose second line",
        "  carries the rest of the sentence",
    ], got


def test_unreleased_lines_keys_on_whole_lines_not_a_bold_title_regex():
    """The extractor takes whole lines rather than keying on a single-line bold
    title, so a bullet whose bold title wraps across lines has its first line
    collected like any other."""
    ch = _load_check_hygiene()
    text = ("## [Unreleased]\n\n### Added\n"
            "- **A title that does not close its bold marker on the first\n"
            "  line at all** and then keeps going.\n")
    assert ch._changelog_unreleased_lines(text) == [
        "- **A title that does not close its bold marker on the first",
        "  line at all** and then keeps going.",
    ]


def test_unreleased_drop_of_a_wrapped_title_bullet_is_reported():
    """Dropping a bullet whose bold title spans two lines is reported, with both
    of its lines."""
    ch = _load_check_hygiene()
    old = ("## [Unreleased]\n\n### Added\n"
           "- **A wrapped title that closes\n  on the second line** with a body.\n"
           "- plain bullet\n")
    new = "## [Unreleased]\n\n### Added\n- plain bullet\n"
    assert ch._changelog_dropped_unreleased_lines(old, new) == [
        "- **A wrapped title that closes",
        "  on the second line** with a body.",
    ]


def test_unreleased_drop_of_sibling_bullet_is_reported():
    """NEGATIVE: a baseline draft bullet missing from the working copy is
    reported, carrying the exact lost line so it can be restored verbatim."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace("- sibling bullet a rebase must not eat\n", "")
    dropped = ch._changelog_dropped_unreleased_lines(_DRAFT_BASE, new)
    assert dropped == ["- sibling bullet a rebase must not eat"], dropped


def test_unreleased_drop_own_added_bullet_is_clean():
    """Adding your own draft bullet (the normal PR flow) must not warn."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace(
        "- my own draft bullet\n",
        "- my own draft bullet\n- a brand new bullet this PR adds\n")
    assert ch._changelog_dropped_unreleased_lines(_DRAFT_BASE, new) == []


def test_unreleased_drop_reworded_bullet_warns_by_design():
    """Rewording a draft line in place IS reported: the check matches exact
    lines, with no similarity heuristic."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace("- my own draft bullet\n",
                              "- my own draft bullet, reworded\n")
    dropped = ch._changelog_dropped_unreleased_lines(_DRAFT_BASE, new)
    assert dropped == ["- my own draft bullet"], dropped


def test_unreleased_drop_release_cut_move_is_clean():
    """Cutting a release MOVES the draft bullets under a new version header. The
    comparison is against the whole working file, not just its [Unreleased]
    section, precisely so a cut does not read as a mass drop."""
    ch = _load_check_hygiene()
    draft_block = (
        "- my own draft bullet\n"
        "- sibling bullet a rebase must not eat\n"
        "- a wrapped draft bullet whose second line\n"
        "  carries the rest of the sentence\n")
    new = _DRAFT_BASE.replace(
        "## [Unreleased]\n\n### Added\n" + draft_block,
        "## [Unreleased]\n\n## [0.2.0] - 2026-08-01\n\n### Added\n" + draft_block)
    assert ch._changelog_dropped_unreleased_lines(_DRAFT_BASE, new) == []


def test_unreleased_drop_continuation_line_loss_is_reported():
    """A wrapped bullet's continuation line is draft content too: a rebase that
    eats just the wrapped half must be reported, same as a whole bullet."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace("  carries the rest of the sentence\n", "")
    dropped = ch._changelog_dropped_unreleased_lines(_DRAFT_BASE, new)
    assert dropped == ["  carries the rest of the sentence"], dropped


def test_unreleased_drop_subsection_header_removal_is_clean():
    """'### Added' inside the draft is scaffolding the draft may freely
    reorganize (e.g. merging two subsections); only content lines are
    watched."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace("### Added\n- my own draft bullet\n",
                              "- my own draft bullet\n")
    assert ch._changelog_dropped_unreleased_lines(_DRAFT_BASE, new) == []


def test_unreleased_drop_no_draft_section_is_clean():
    """A baseline with no [Unreleased] section at all has nothing to watch."""
    ch = _load_check_hygiene()
    old = ("# Changelog\n\n## [0.1.0] - 2026-07-04\n\n### Added\n"
           "- inference and CLI\n")
    assert ch._changelog_dropped_unreleased_lines(old, "# Changelog\n") == []


def test_unreleased_drop_clamps_to_the_draft_count():
    """The lost-count is clamped to how many copies the DRAFT actually held, so a
    line duplicated across the draft AND a published section reports one loss, not
    two, when both copies go.

    Pins the `min(...)` in _changelog_dropped_unreleased_lines; an unclamped
    subtraction reports the line twice."""
    ch = _load_check_hygiene()
    old = _DRAFT_BASE.replace("- my own draft bullet\n", "- inference and CLI\n")
    new = old.replace("- inference and CLI\n", "")          # BOTH copies removed
    dropped = ch._changelog_dropped_unreleased_lines(old, new)
    assert dropped == ["- inference and CLI"], dropped


def test_unreleased_drop_published_duplicate_does_not_mask():
    """A draft line whose text ALSO appears in a published section must still be
    reported when the DRAFT copy is deleted: occurrences are counted across the
    whole file on both sides, so the surviving published copy cannot satisfy
    the draft copy's count."""
    ch = _load_check_hygiene()
    old = _DRAFT_BASE.replace("- my own draft bullet\n",
                              "- inference and CLI\n")   # duplicates 0.1.0's bullet
    new = old.replace("- inference and CLI\n", "", 1)    # deletes the DRAFT copy only
    dropped = ch._changelog_dropped_unreleased_lines(old, new)
    assert dropped == ["- inference and CLI"], dropped


def test_unreleased_drop_guard_warns_end_to_end(tmp_path, monkeypatch):
    """The git-wired entrypoint: with the baseline committed, dropping a sibling
    draft bullet in the working tree yields ONE warning naming the lost line,
    while the hard append-only gate stays silent, since the draft is exempt from
    it."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_changelog_repo(tmp_path, _DRAFT_BASE)
    assert ch._changelog_unreleased_drops() == []          # no change -> quiet

    (tmp_path / "CHANGELOG.md").write_text(
        _DRAFT_BASE.replace("- sibling bullet a rebase must not eat\n",
                            "- a bullet this branch adds\n"),
        encoding="utf-8")
    msgs = ch._changelog_unreleased_drops()
    assert len(msgs) == 1, msgs
    assert "sibling bullet a rebase must not eat" in msgs[0]
    assert "my own draft bullet" not in msgs[0]            # untouched line: not listed
    assert ch._changelog_append_only() == []               # the hard gate cannot see it


def test_unreleased_new_duplicate_bullet_is_reported():
    """NEGATIVE: a bullet that appears twice in the working copy is reported,
    with its working-copy count."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace(
        "- my own draft bullet\n",
        "- my own draft bullet\n- sibling bullet a rebase must not eat\n")
    dupes = ch._changelog_new_duplicate_unreleased_bullets(_DRAFT_BASE, new)
    assert dupes == [("- sibling bullet a rebase must not eat", 2)], dupes


def test_unreleased_duplicate_already_in_the_baseline_is_not_reported():
    """A duplicate that already exists at the baseline is not reported."""
    ch = _load_check_hygiene()
    old = _DRAFT_BASE.replace("- my own draft bullet\n",
                              "- my own draft bullet\n- my own draft bullet\n")
    assert ch._changelog_new_duplicate_unreleased_bullets(old, old) == []


def test_unreleased_duplicate_growing_past_the_baseline_is_reported():
    """A duplicate that grows past the baseline (two copies there, three in the
    working copy) IS reported."""
    ch = _load_check_hygiene()
    old = _DRAFT_BASE.replace("- my own draft bullet\n",
                              "- my own draft bullet\n- my own draft bullet\n")
    new = old.replace("- my own draft bullet\n- my own draft bullet\n",
                      "- my own draft bullet\n- my own draft bullet\n"
                      "- my own draft bullet\n")
    assert ch._changelog_new_duplicate_unreleased_bullets(old, new) == [
        ("- my own draft bullet", 3)]


def test_unreleased_duplicate_ignores_continuation_lines():
    """Only top-level bullets are counted: two DIFFERENT bullets can legitimately
    wrap to the same trailing words, so an identical continuation line is not a
    duplicate. Two byte-identical bullet lines are a mistake every time."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace(
        "  carries the rest of the sentence\n",
        "  carries the rest of the sentence\n"
        "- another bullet whose wrap repeats\n"
        "  carries the rest of the sentence\n")
    assert ch._changelog_new_duplicate_unreleased_bullets(_DRAFT_BASE, new) == []


def test_unreleased_duplicate_outside_the_draft_is_ignored():
    """A published section is the append-only gate's business, not this one's:
    the duplicate check looks only inside [Unreleased]."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace("- inference and CLI\n",
                              "- inference and CLI\n- inference and CLI\n")
    assert ch._changelog_new_duplicate_unreleased_bullets(_DRAFT_BASE, new) == []


def test_unreleased_duplicate_guard_warns_end_to_end(tmp_path, monkeypatch):
    """The git-wired duplicate entrypoint on a real throwaway repo: quiet on the
    committed baseline, one warning naming the doubled bullet after the bad
    restore, and the DROP check stays quiet (nothing was lost) so the two
    conditions are reported independently."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_changelog_repo(tmp_path, _DRAFT_BASE)
    assert ch._changelog_unreleased_duplicates() == []

    (tmp_path / "CHANGELOG.md").write_text(
        _DRAFT_BASE.replace(
            "- my own draft bullet\n",
            "- my own draft bullet\n- sibling bullet a rebase must not eat\n"),
        encoding="utf-8")
    msgs = ch._changelog_unreleased_duplicates()
    assert len(msgs) == 1, msgs
    assert "sibling bullet a rebase must not eat" in msgs[0]
    assert "x2" in msgs[0], msgs[0]
    assert ch._changelog_unreleased_drops() == []


def test_hygiene_main_warns_on_a_new_duplicate(tmp_path, monkeypatch, capsys):
    """main() surfaces the duplicate condition too (warn by default, escalated by
    --strict) - a second detector wired into the same warning channel."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])
    # Stub the release-manifest gate as well as the file scans: it reads scripts/
    # next to check_hygiene.py on real disk, which the REPO monkeypatch does not
    # redirect, and its warning would move these exit codes and issue counts.
    monkeypatch.setattr(ch, "_release_manifest_gate", lambda: ([], []))
    monkeypatch.delenv("LOCALM_HYGIENE_STRICT", raising=False)
    _init_changelog_repo(tmp_path, _DRAFT_BASE)
    (tmp_path / "CHANGELOG.md").write_text(
        _DRAFT_BASE.replace(
            "- my own draft bullet\n",
            "- my own draft bullet\n- sibling bullet a rebase must not eat\n"),
        encoding="utf-8")

    assert ch.main([]) == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "more than once" in err, err
    assert ch.main(["--strict"]) == 1


def test_unreleased_drop_guard_passes_without_a_git_baseline(tmp_path, monkeypatch):
    """A never-committed CHANGELOG has no baseline: no warning, no crash."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "CHANGELOG.md").write_text(_DRAFT_BASE, encoding="utf-8")
    assert ch._changelog_unreleased_drops() == []


def test_hygiene_main_warns_but_passes_on_a_draft_drop(tmp_path, monkeypatch, capsys):
    """main() end to end on a real throwaway repo: a dropped draft bullet prints
    a WARNING and still exits 0 by default; --strict and LOCALM_HYGIENE_STRICT=1
    each escalate the same run to a failure; strict with nothing dropped stays 0
    (the knob escalates warnings, it does not invent them)."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])
    # Stub the release-manifest gate as well as the file scans: it reads scripts/
    # next to check_hygiene.py on real disk, which the REPO monkeypatch does not
    # redirect, and its warning would move these exit codes and issue counts.
    monkeypatch.setattr(ch, "_release_manifest_gate", lambda: ([], []))
    monkeypatch.delenv("LOCALM_HYGIENE_STRICT", raising=False)
    _init_changelog_repo(tmp_path, _DRAFT_BASE)

    assert ch.main(["--strict"]) == 0          # strict on a clean tree: still 0

    (tmp_path / "CHANGELOG.md").write_text(
        _DRAFT_BASE.replace("- sibling bullet a rebase must not eat\n", ""),
        encoding="utf-8")

    assert ch.main([]) == 0                    # warn-only by default
    err = capsys.readouterr().err
    assert "WARNING" in err, err
    assert "sibling bullet a rebase must not eat" in err, err

    assert ch.main(["--strict"]) == 1          # the flag escalates
    err = capsys.readouterr().err
    assert "FAILED" in err, err
    assert "sibling bullet a rebase must not eat" in err, err

    monkeypatch.setenv("LOCALM_HYGIENE_STRICT", "1")
    assert ch.main([]) == 1                    # the env knob escalates
    monkeypatch.setenv("LOCALM_HYGIENE_STRICT", "0")
    assert ch.main([]) == 0                    # explicit off stays warn-only


def test_baseline_ref_is_pinned_once_per_run(monkeypatch, tmp_path):
    """The baseline sha must be resolved ONCE per process, not re-resolved by each
    check. origin/master is a MOVING ref (worktrees share one ref store, so a
    sibling's fetch or a landing merge can advance it mid-run), and the three
    callers - the append-only gate, the [Unreleased] warn-only checks, and the
    service-worker cache-bump gate - would otherwise compare against DIFFERENT
    baselines within one run. Asserted by counting git invocations."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    ch._BASELINE_REF_CACHE.clear()
    calls = []

    class _Result:
        returncode = 0
        stdout = "feedfacefeedfacefeedfacefeedfacefeedface\n"

    def _fake_git(*args):
        calls.append(args)
        return _Result()

    monkeypatch.setattr(ch, "_git", _fake_git)
    first = ch._changelog_baseline_ref()
    for _ in range(5):
        assert ch._changelog_baseline_ref() == first
    merge_base_calls = [c for c in calls if c[:1] == ("merge-base",)]
    assert len(merge_base_calls) == 1, merge_base_calls


def test_baseline_ref_cache_is_keyed_on_repo(monkeypatch, tmp_path):
    """Pinning must not leak ACROSS trees: repointing REPO (what every test here
    does, and what a caller embedding this module could do) resolves afresh rather
    than reusing another repo's sha."""
    ch = _load_check_hygiene()
    ch._BASELINE_REF_CACHE.clear()
    shas = iter(["a" * 40, "b" * 40])
    seen = []

    def _fake_git(*args):
        class _R:
            returncode = 0
            stdout = next(shas) + "\n"
        seen.append(args)
        return _R()

    monkeypatch.setattr(ch, "_git", _fake_git)
    monkeypatch.setattr(ch, "REPO", tmp_path / "one")
    assert ch._changelog_baseline_ref() == "a" * 40
    monkeypatch.setattr(ch, "REPO", tmp_path / "two")
    assert ch._changelog_baseline_ref() == "b" * 40


# ---- report-only: which [Unreleased] bullets does THIS branch add? -----------
# Lists the draft bullets the working copy adds over the baseline.

def test_added_unreleased_bullets_lists_only_new_ones():
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace("- my own draft bullet\n",
                              "- my own draft bullet\n- a bullet this branch adds\n")
    assert ch._changelog_added_unreleased_bullets(_DRAFT_BASE, new) == [
        "- a bullet this branch adds"]


def test_added_unreleased_bullets_counts_an_extra_copy_as_added():
    """A second copy of an existing bullet is an addition too (and is separately
    reported as a duplicate) - so the count cannot be gamed by duplicating."""
    ch = _load_check_hygiene()
    new = _DRAFT_BASE.replace("- my own draft bullet\n",
                              "- my own draft bullet\n- my own draft bullet\n")
    assert ch._changelog_added_unreleased_bullets(_DRAFT_BASE, new) == [
        "- my own draft bullet"]


def test_added_unreleased_bullets_ignores_continuations_and_published():
    """Neither a new CONTINUATION line in the draft nor a new bullet in a PUBLISHED
    section counts as an added draft bullet: only top-level [Unreleased] bullets do."""
    ch = _load_check_hygiene()
    new = (_DRAFT_BASE
           .replace("- inference and CLI\n",
                    "- inference and CLI\n- a published-section bullet\n")
           .replace("  carries the rest of the sentence\n",
                    "  carries the rest of the sentence\n  and now a third line\n"))
    assert ch._changelog_added_unreleased_bullets(_DRAFT_BASE, new) == []


def test_added_note_is_report_only_and_rides_along_with_a_warning(
        tmp_path, monkeypatch, capsys):
    """The added-bullet report is CONTEXT, never its own warning: a run that only
    ADDS bullets stays completely quiet (this gate is a pre-commit hook), but once
    a real warning fires the note rides along so the reader can tell which bullets
    are theirs."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])
    # Stub the release-manifest gate as well as the file scans: it reads scripts/
    # next to check_hygiene.py on real disk, which the REPO monkeypatch does not
    # redirect, and its warning would move these exit codes and issue counts.
    monkeypatch.setattr(ch, "_release_manifest_gate", lambda: ([], []))
    monkeypatch.delenv("LOCALM_HYGIENE_STRICT", raising=False)
    _init_changelog_repo(tmp_path, _DRAFT_BASE)

    # add-only: quiet, and NOT escalated by --strict either
    (tmp_path / "CHANGELOG.md").write_text(
        _DRAFT_BASE.replace("- my own draft bullet\n",
                            "- my own draft bullet\n- purely additive bullet\n"),
        encoding="utf-8")
    ch._BASELINE_REF_CACHE.clear()
    assert ch.main(["--strict"]) == 0
    assert "for context" not in capsys.readouterr().err

    # a real drop alongside an addition: warning + the context note
    (tmp_path / "CHANGELOG.md").write_text(
        _DRAFT_BASE.replace("- sibling bullet a rebase must not eat\n",
                            "- purely additive bullet\n"),
        encoding="utf-8")
    ch._BASELINE_REF_CACHE.clear()
    assert ch.main([]) == 0
    err = capsys.readouterr().err
    assert "sibling bullet a rebase must not eat" in err
    assert "for context, this branch adds 1 [Unreleased] bullet(s)" in err, err
    assert "added: '- purely additive bullet'" in err, err


def test_added_note_does_not_inflate_the_strict_failure_count(
        tmp_path, monkeypatch, capsys):
    """Under --strict the report-only note is FOLDED INTO the warning it
    accompanies, not counted as a hygiene issue of its own."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])
    # Stub the release-manifest gate as well as the file scans: it reads scripts/
    # next to check_hygiene.py on real disk, which the REPO monkeypatch does not
    # redirect, and its warning would move these exit codes and issue counts.
    monkeypatch.setattr(ch, "_release_manifest_gate", lambda: ([], []))
    monkeypatch.delenv("LOCALM_HYGIENE_STRICT", raising=False)
    _init_changelog_repo(tmp_path, _DRAFT_BASE)
    (tmp_path / "CHANGELOG.md").write_text(
        _DRAFT_BASE.replace("- sibling bullet a rebase must not eat\n", ""),
        encoding="utf-8")

    assert ch.main(["--strict"]) == 1
    err = capsys.readouterr().err
    assert "1 hygiene issue(s)" in err, err
    assert "2 hygiene issue(s)" not in err, err
    assert "for context" in err, err      # still shown, just not counted


def test_strict_env_knob_off_values(monkeypatch):
    """The env knob's off-set is explicit: empty/0/false/no/off (any case, any
    surrounding whitespace) stay warn-only. Anything ELSE means strict, so a
    typo fails toward MORE checking rather than silently disabling the gate."""
    ch = _load_check_hygiene()
    for off in ("", "0", "false", "FALSE", "no", "off", "OFF", "  off  "):
        monkeypatch.setenv("LOCALM_HYGIENE_STRICT", off)
        assert ch._strict_env() is False, off
    for on in ("1", "true", "yes", "on", "strict", "please"):
        monkeypatch.setenv("LOCALM_HYGIENE_STRICT", on)
        assert ch._strict_env() is True, on
    monkeypatch.delenv("LOCALM_HYGIENE_STRICT", raising=False)
    assert ch._strict_env() is False


def test_real_changelog_unreleased_checks_run_on_the_real_tree():
    """Smoke: both detectors run against the REAL repo without crashing.
    Emptiness is NOT asserted, so a branch that rewords an [Unreleased] line
    stays warn-only rather than failing the suite."""
    ch = _load_check_hygiene()
    assert isinstance(ch._changelog_unreleased_drops(), list)
    assert isinstance(ch._changelog_unreleased_duplicates(), list)


def test_real_changelog_has_no_duplicate_unreleased_bullets():
    """The real [Unreleased] section must have no duplicate bullets AT ALL (not
    just none newly introduced).

    Unlike the drop check above, emptiness IS asserted here: if master ever
    lands a duplicate, unrelated PRs go red until someone removes it."""
    ch = _load_check_hygiene()
    from collections import Counter
    text = (ch.REPO / ch._CHANGELOG).read_text(encoding="utf-8")
    bullets = Counter(x for x in ch._changelog_unreleased_lines(text)
                      if x.startswith("- "))
    assert [b for b, n in bullets.items() if n > 1] == []


def test_baseline_ref_is_the_merge_base_not_the_moving_tip(monkeypatch):
    """The baseline MUST be the merge-base with origin/master, never the
    origin/master ref itself. Worktrees share one ref store, so a sibling
    session's fetch advances the tip, and a tip-relative comparison then reports
    every bullet merged after the branch point as a dropped line. Asserted at
    the git-command level."""
    ch = _load_check_hygiene()
    calls = []

    class _Result:
        returncode = 0
        stdout = "cafebabecafebabecafebabecafebabecafebabe\n"

    def _fake_git(*args):
        calls.append(args)
        return _Result()

    monkeypatch.setattr(ch, "_git", _fake_git)
    ref = ch._changelog_baseline_ref()
    assert calls and calls[0] == ("merge-base", "HEAD", "origin/master"), calls
    assert ref == "cafebabecafebabecafebabecafebabecafebabe"
