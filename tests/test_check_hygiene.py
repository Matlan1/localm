# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py must scan .html for machine-absolute paths.

The absolute-path heuristic (check 3) was gated on _CODE_EXTS, which omitted
.html, so a drive-letter placeholder in index.html slipped through. These tests
pin that .html is scanned and that the shipped index.html is clean.
"""

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_html_drive_letter_is_detected(tmp_path, monkeypatch):
    """NEGATIVE: a drive-letter path in an .html must be flagged.

    Pre-fix .html is not in _CODE_EXTS so _scan returns no absolute-path
    problem and this fails; post-fix it is flagged.
    """
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
    """Regression guard: the real GUI index.html must stay free of drive-letter
    placeholders now that .html is in scope."""
    ch = _load_check_hygiene()
    index = REPO_ROOT / "localm" / "plugins" / "gui" / "static" / "index.html"
    problems = [x for x in ch._scan(index) if "absolute/machine path" in x]
    assert not problems, problems


def test_html_is_in_code_exts():
    ch = _load_check_hygiene()
    assert ".html" in ch._CODE_EXTS


# ---- check 2: secret disclosure --------------------------------------------
# The synthetic tokens below are assembled from fragments at runtime so this
# test file does not itself contain a literal secret that the hygiene check
# would (correctly) flag when it scans the tracked tree.

def test_finegrained_github_pat_is_detected(tmp_path, monkeypatch):
    """NEGATIVE: a modern fine-grained GitHub PAT (github_pat_11...) must be
    flagged as a disclosure. The old check only knew the classic ghp_ prefix,
    so a committed fine-grained PAT would have slipped through."""
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
# The changelog is the permanent public record of what shipped: a release ADDS a
# section on top; existing entries are never deleted or rewritten. The guard diffs
# the working CHANGELOG against the last committed version and fails if any shipped
# entry line disappeared. Headers and link-reference definitions are exempt (they
# legitimately change when a release is cut).

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
    """NEGATIVE (the confirmed gap): the guard skipped every '## ' line when building
    the protected set, including the version header ITSELF, so silently changing a
    published release's ship date in place went undetected. The header carries real
    shipped-record content (version + date), same as any body entry."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace(
        "## [0.1.0] - 2026-07-04", "## [0.1.0] - 2026-07-09")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["## [0.1.0] - 2026-07-04"], removed


def test_changelog_renumbering_a_published_version_header_fails():
    """Same gap, the other direction: silently renumbering a shipped version (e.g.
    passing off 0.1.0 as 0.1.1) must be caught too."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace(
        "## [0.1.0] - 2026-07-04", "## [0.1.1] - 2026-07-04")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["## [0.1.0] - 2026-07-04"], removed


def test_changelog_renaming_a_published_subsection_header_fails():
    """NEGATIVE (a second confirmed gap, same mechanism): '### Added' etc. inside an
    already-published section was never in the protected set (the 'stripped.
    startswith(\"#\")' catch-all excluded it unconditionally), so flipping it to
    '### Removed' - misrepresenting what a published release shipped, while leaving
    the bullet text unchanged - went undetected. Reproduces the reviewer's exact
    repro: pre-fix this returned []."""
    ch = _load_check_hygiene()
    new = _BASE_CHANGELOG.replace("### Added\n- inference and CLI\n- the GUI",
                                  "### Removed\n- inference and CLI\n- the GUI")
    removed = ch._changelog_removed_lines(_BASE_CHANGELOG, new)
    assert removed == ["### Added"], removed


def test_changelog_deleting_a_published_subsection_header_fails():
    """Same gap, the other reviewer-confirmed attack: deleting '### Added' entirely
    (orphaning its bullets directly under the version header) must also be caught."""
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
    """The git-wired entrypoint (the actual CI/pre-commit-hook gate) must catch a
    header-only rewrite of a published section too - reproduces the confirmed gap end
    to end, not just at the pure _changelog_removed_lines layer."""
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


def test_real_changelog_is_append_only_against_head():
    """The real repo CHANGELOG must itself satisfy the guard (this PR only adds a
    header note on top, removing nothing)."""
    ch = _load_check_hygiene()
    assert ch._changelog_append_only() == []
