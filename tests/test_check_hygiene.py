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


# ---- check 3: .js/.mjs/.yaml/.yml coverage (LM-DA-022) ----------------------
# The absolute-path heuristic (check 3) was gated on _CODE_EXTS, which omitted
# .js/.mjs/.yaml/.yml, so a hardcoded drive-letter path in the GUI's
# hand-written JS frontend (or a workflow/config YAML file) would slip
# through, even though the em-dash and disclosure checks already covered
# these extensions. These tests pin that .js/.mjs/.yaml/.yml are scanned too.

def test_js_absolute_path_is_detected(tmp_path, monkeypatch):
    """NEGATIVE: a drive-letter path in a .js file must be flagged.

    Pre-fix .js is not in _CODE_EXTS so _scan returns no absolute-path
    problem and this fails; post-fix it is flagged.
    """
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
    p.write_text('path: "C:\\\\Users\\\\someone\\\\build"\n', encoding="utf-8")
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
    """The Python is_test heuristic (tests/, test_*.py) doesn't match the
    frontend suites' own conventions (tests-js/*.test.mjs, tests-e2e/*.spec.mjs),
    which legitimately use synthetic absolute paths as mock fixtures - these
    must stay exempt now that .mjs is in scope, the same way tests/test_*.py
    already is."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    for rel in ("tests-js/upload.test.mjs", "tests-e2e/boot-and-click.spec.mjs"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('const dir = "/home/uploads";\n', encoding="utf-8")
        problems = ch._scan(p)
        assert not [x for x in problems if "absolute/machine path" in x], (rel, problems)


def test_shipped_frontend_js_has_no_machine_paths():
    """Regression guard: the real GUI/CLI JS files must stay free of drive-letter
    or /home//Users absolute paths now that .js/.mjs are in scope."""
    ch = _load_check_hygiene()
    tracked = ch._tracked_files()
    problems = []
    for f in tracked:
        if f.suffix.lower() in (".js", ".mjs"):
            problems.extend(x for x in ch._scan(f) if "absolute/machine path" in x)
    assert not problems, problems


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


def test_git_helper_decodes_output_as_utf8_not_locale_default(monkeypatch):
    """REGRESSION: _git() must decode git's output as UTF-8, not the platform
    locale default.

    The changelog append-only guard reads the baseline via `git show
    <ref>:CHANGELOG.md` (through _git) and the working tree via
    read_text(encoding="utf-8"). With a bare text=True, Python decodes the
    subprocess output with locale.getpreferredencoding() - cp1252 on Windows -
    so a UTF-8 line carrying an emoji or an accented character comes back as
    mojibake and no longer equals the UTF-8 working-tree line, falsely flagging
    an unchanged PUBLISHED entry as a rewrite. Pinning encoding="utf-8" is what
    makes the two reads match on every platform; assert it here so the kwarg is
    never silently dropped (a behavioral test would only bite on a cp1252 box,
    not on CI Linux)."""
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


def test_changelog_append_only_no_false_positive_on_non_ascii_under_cp1252(
        tmp_path, monkeypatch):
    """A shipped entry with NON-ASCII content (an emoji, an accented word) must
    not be reported as removed just because the process locale is not UTF-8.

    Regression for the Windows false positive: ``_git`` compares
    ``git show <ref>:CHANGELOG.md`` against the working file read as UTF-8. git
    emits UTF-8, but ``subprocess(text=True)`` alone decodes with the locale
    codepage - cp1252 on Windows - turning every non-ASCII shipped line into
    mojibake that no longer matched its own UTF-8 working copy, so the guard
    flagged a byte-identical CHANGELOG. Forcing a cp1252 locale reproduces it on
    ANY platform (Linux CI decodes UTF-8 by default and never saw it); the fix
    is ``_git`` decoding explicitly as UTF-8, which this monkeypatch cannot
    perturb."""
    import locale
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    # A shipped entry line carrying both an emoji and an accented word - exactly
    # the shapes the real 0.1.2 changelog carried (a memory-chip glyph, an
    # accented unicode-key example) that triggered the false failure.
    non_ascii = _BASE_CHANGELOG.replace(
        "- the GUI\n",
        "- the GUI, now with a \U0001f9e0 memory chip and an äccented note\n")
    _init_changelog_repo(tmp_path, non_ascii)
    # working tree is byte-identical to HEAD -> nothing removed, on any locale.
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "cp1252")
    assert ch._changelog_append_only() == []
    # And the raw git read round-trips the non-ASCII intact (the decode itself,
    # independent of the diff layer above).
    got = ch._git("show", "HEAD:CHANGELOG.md")
    assert got is not None and got.returncode == 0
    assert "\U0001f9e0" in got.stdout and "äccented" in got.stdout


def test_real_changelog_is_append_only_against_head():
    """The real repo CHANGELOG must itself satisfy the guard (this PR only adds a
    header note on top, removing nothing)."""
    ch = _load_check_hygiene()
    assert ch._changelog_append_only() == []


# ---- the pending (cut but UNRELEASED) section is a draft, not history --------
# "## [x.y.z]" alone does not mean x.y.z shipped: the release ritual bumps VERSION and
# cuts the section BEFORE the tag exists. Freezing that draft protects nothing (nobody
# can have downloaded a release that does not exist) and blocks the legitimate final
# cut - re-dating the section on the day it really ships, or folding newer [Unreleased]
# work into a prep that was never published. These pin BOTH directions: the pending
# section is editable, and everything else stays frozen even when tags are absent.

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
# _raw_accessor_violations enforces that a "single -> combined N resources"
# capability (vram_info() -> vram_capacity() is the first case) cannot be
# silently bypassed by a future call site re-adopting the raw single-resource
# accessor. Written after a fresh-context review of the fix demonstrated that
# the FIRST version of this check had a real, working bypass (an import
# alias) - these tests pin both the direct-call and alias-call detection so a
# future edit that silently weakens the check (e.g. dropping the
# ast.ImportFrom alias-tracking) fails a test, not just "gets noticed later".

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
    """NEGATIVE (the confirmed gap): `from localm.discover import vram_info as
    vi` followed by `vi()` must ALSO be flagged - a fresh-context review
    demonstrated this alias genuinely evaded the first version of this check
    (which matched only the literal name `vram_info`, missing that an
    ast.ImportFrom alias rebinds it locally)."""
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
    """`import localm.discover as disc; disc.vram_info()` must be flagged too -
    ast.Attribute.attr is still the literal accessor name no matter what the
    MODULE is aliased to, so this path never needed the alias-tracking fix."""
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


# ---- HONESTY FIX #5: an unparseable .py must be REPORTED, not silently skipped ----
# The raw-accessor guard used to `continue` past any .py it could not read or parse
# (OSError / SyntaxError / UnicodeDecodeError), so a file it never actually scanned
# read as "clean" (AGENTS.md rule 5). It now appends a problem instead.

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
    """Positive control for FIX #5: a parseable .py with no raw-accessor call stays
    clean - the new report path fires ONLY on a genuinely unparseable file."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    p = tmp_path / "fine.py"
    p.write_text("def f():\n    return 1 + 2\n", encoding="utf-8")
    assert ch._raw_accessor_violations([p]) == []


# ---- HONESTY FIX #3: the top-level gate must FAIL when it scanned nothing ----
# When `git ls-files` fails (not a checkout, git missing), _tracked_files() returns []
# so the dash/disclosure/abs-path scan and changelog gate run over ZERO files and
# main() used to print "Hygiene check passed". A disclosure/privacy gate reporting
# clean without scanning anything is exactly the silent pass rule 5 forbids.

def test_hygiene_main_fails_when_nothing_scanned(monkeypatch):
    """NEGATIVE: with no tracked files enumerable (simulated git failure), main() must
    return non-zero and NOT report success."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "_tracked_files", lambda: [])
    assert ch.main([]) == 1


def test_hygiene_main_passes_when_files_are_scanned(tmp_path, monkeypatch):
    """Positive control for FIX #3: with a real (clean) file to scan and the other
    sub-gates stubbed clean, main() returns 0 - the new guard fires ONLY on the
    scanned-nothing case, it does not break the happy path."""
    ch = _load_check_hygiene()
    clean = tmp_path / "ok.py"
    clean.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(ch, "REPO", tmp_path)
    monkeypatch.setattr(ch, "_tracked_files", lambda: [clean])
    monkeypatch.setattr(ch, "_changelog_append_only", lambda: [])
    monkeypatch.setattr(ch, "_raw_accessor_violations", lambda t: [])
    monkeypatch.setattr(ch, "_manifest_problems", lambda: [])
    assert ch.main([]) == 0


# ---- [Unreleased] draft drop WARNING (rebase-eats-a-sibling-bullet backstop) --
# The append-only gate above deliberately EXEMPTS the [Unreleased] draft: it is
# freely rewritable until cut (test_changelog_rewriting_unreleased_is_allowed).
# But that exemption has a blind spot: when parallel branches all add draft
# bullets, a textually clean rebase can mis-anchor a replayed insertion inside
# the bullet list and silently DELETE a sibling branch's bullet - seen twice in
# one 12-PR fan-out day (2026-07-22): the rebase reported clean, the hygiene
# gate passed, and a landed PR's entry vanished from the release notes with no
# mechanical backstop. These tests pin the warn-only detector for that: a
# baseline [Unreleased] content line missing from the working copy WARNS (exit
# stays 0), and fails only under --strict or LOCALM_HYGIENE_STRICT=1.

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


def test_unreleased_drop_of_sibling_bullet_is_reported():
    """NEGATIVE (the incident): a baseline draft bullet missing from the working
    copy is reported, carrying the exact lost line so it can be restored
    verbatim."""
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
    """Rewording a draft line in place IS reported - a DOCUMENTED accepted cost,
    not a bug. Exact-line matching is deliberate: a similarity heuristic that
    suppressed near-matches could suppress exactly the incident case (a
    sibling's bullet eaten while similar sibling bullets remain). For a
    warn-only check a false positive costs one glance at the warning; a false
    negative defeats the whole backstop."""
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
    reorganize (e.g. merging two subsections); only content lines are watched.
    Warning on scaffolding would be noise that trains people to ignore the one
    warning that matters."""
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
    while the hard append-only gate stays silent (the draft is exempt from it,
    which is exactly why this warning exists)."""
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
    monkeypatch.setattr(ch, "_manifest_problems", lambda: [])
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


def test_real_changelog_unreleased_drops_runs_on_the_real_tree():
    """Smoke: the drop detector runs against the REAL repo without crashing.
    Deliberately NOT asserting emptiness: a branch that legitimately rewords an
    [Unreleased] line would then fail the SUITE, turning the designed warn-only
    behavior back into a hard failure through the back door."""
    ch = _load_check_hygiene()
    assert isinstance(ch._changelog_unreleased_drops(), list)
