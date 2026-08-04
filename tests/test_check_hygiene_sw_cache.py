# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py's PWA service-worker cache-version bump gate (check 6).

The gate exists because this shipped THREE times undetected by human review: v49 for
#621's settings.js fix, again for the managed_comfy_enabled checkbox removal, and PR
#640's models.js + knowledge.js (a live field bug only by luck - a later unrelated PR
bumped the cache). See check_hygiene.py's own block comment above _SW_STATIC.

These tests are written to survive MUTATION. Asserting `== []` on a clean tree proves
nothing on its own - a gate hardwired to `return []` passes every such test. So every
"clean" assertion here is paired with a positive control that must FAIL, and each
silent-skip path (unparseable CACHE, unparseable SHELL, a moved sw.js) is asserted to
be LOUD rather than merely absent. A gate that cannot fail is decoration.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_ENV = {**os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Mirrors the real sw.js's shape: a CACHE constant, a SHELL precache list, and - the
# part that matters - assets that are NOT in SHELL but are still runtime-cached by the
# fetch handler into the same versioned cache (jsQR and the fonts, as in production).
_SW_JS_TEXT = (
    'const CACHE = "localm-shell-v1";\n'
    'const SHELL = [\n'
    '  "/", "/index.html", "/style.css",\n'
    '  "/app/main.js",\n'
    '  "/pages/settings.js", "/pages/models.js",\n'
    '];\n'
)

_STATIC = "localm/plugins/gui/static"


def _git(tmp_path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=tmp_path, check=True, env=_ENV,
                   capture_output=True)


def _init_fake_repo(tmp_path: Path) -> Path:
    """A repo shaped like localm's static tree. Returns the static root."""
    _git(tmp_path, "init", "-q")
    static = tmp_path / _STATIC
    (static / "pages").mkdir(parents=True)
    (static / "app").mkdir(parents=True)
    (static / "vendor" / "fonts").mkdir(parents=True)
    (static / "sw.js").write_text(_SW_JS_TEXT, encoding="utf-8")
    (static / "index.html").write_text("<html></html>", encoding="utf-8")
    (static / "style.css").write_text("body {}", encoding="utf-8")
    (static / "app" / "main.js").write_text("// main v1\n", encoding="utf-8")
    (static / "pages" / "settings.js").write_text("// settings v1\n", encoding="utf-8")
    (static / "pages" / "models.js").write_text("// models v1\n", encoding="utf-8")
    # Runtime-cached but NOT in SHELL - the production blind spot (jsQR + KaTeX fonts).
    (static / "vendor" / "jsQR.min.js").write_text("// jsQR v1\n", encoding="utf-8")
    (static / "vendor" / "fonts" / "KaTeX_Main-Regular.woff2").write_bytes(b"font-v1")
    (tmp_path / "README.md").write_text("not a static asset\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "baseline")
    return static


def _bump(static: Path, to: str = "v2") -> None:
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace(
        'const CACHE = "localm-shell-v1"', f'const CACHE = "localm-shell-{to}"'),
        encoding="utf-8")


# ---- the core contract ----------------------------------------------------------

def test_no_change_is_clean(tmp_path, monkeypatch):
    """Happy path: an untouched checkout passes. Paired with the positive controls
    below, which prove this is a real [] and not a hardwired one."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_fake_repo(tmp_path)
    assert ch._sw_cache_bump_violations() == []


def test_precached_file_changed_without_cache_bump_fails(tmp_path, monkeypatch):
    """The exact regression this gate exists for (#640): a precached file changes,
    CACHE does not. Must flag, and must name the file."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "settings.js").write_text("// v2 real change\n", encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert problems, "changing a precached file without bumping CACHE must be flagged"
    assert any("pages/settings.js" in p and "CACHE" in p for p in problems)


def test_precached_file_changed_with_cache_bump_is_clean(tmp_path, monkeypatch):
    """The correct fix: bumping CACHE alongside the change clears the flag."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "settings.js").write_text("// v2 real change\n", encoding="utf-8")
    _bump(static)

    assert ch._sw_cache_bump_violations() == []


def test_non_static_file_changing_is_not_flagged(tmp_path, monkeypatch):
    """No false positives: a file the service worker cannot cache (outside the static
    tree) is none of this gate's business."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_fake_repo(tmp_path)
    (tmp_path / "README.md").write_text("edited, but not a cacheable asset\n",
                                        encoding="utf-8")

    assert ch._sw_cache_bump_violations() == []


# ---- the blind spot the SHELL-only watch set left open --------------------------

def test_runtime_cached_asset_outside_shell_is_flagged(tmp_path, monkeypatch):
    """SHELL is only the PREcache. sw.js's fetch handler runtime-caches every
    same-origin static GET into the SAME versioned cache and serves it cache-first
    forever, so a non-SHELL asset goes stale exactly as hard as a SHELL one. Watching
    SHELL alone left /vendor/jsQR.min.js (lazily loaded by the QR scanner) and all 20
    KaTeX fonts permanently unwatched in production."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "vendor" / "jsQR.min.js").write_text("// jsQR v2 decode fix\n",
                                                   encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert problems, "a runtime-cached asset outside SHELL must still require a bump"
    assert any("jsQR.min.js" in p for p in problems)


def test_binary_asset_outside_shell_is_flagged(tmp_path, monkeypatch):
    """A .woff2 under vendor/ is doubly invisible to main()'s _tracked_files() (it
    drops _BINARY_EXTS and skips vendor/ wholesale), so this check must enumerate the
    static tree itself rather than reuse that list."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "vendor" / "fonts" / "KaTeX_Main-Regular.woff2").write_bytes(b"font-v2")

    problems = ch._sw_cache_bump_violations()

    assert problems, "a changed precacheable font must require a bump"
    assert any("KaTeX_Main-Regular.woff2" in p for p in problems)


def test_staged_deletion_of_precached_file_is_flagged(tmp_path, monkeypatch):
    """A DELETED asset is a change too, and the deletion must be STAGED (`git rm`) to
    match reality: the pre-commit hook runs after `git add`, and CI diffs a committed
    HEAD. An earlier version filtered the diff against `git ls-files` (the INDEX), which
    silently dropped every staged or committed deletion - i.e. every deletion a real
    invocation produces. The unstaged-unlink variant below is the ONLY case that passed
    then, which is why testing it alone was false confidence."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    _git(tmp_path, "rm", "-q", f"{_STATIC}/pages/models.js")
    # Drop it from SHELL too, as a real removal would - the gate must still notice.
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace(', "/pages/models.js"', ""),
                  encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert problems, "a STAGED deletion of a cached asset without a bump must be flagged"
    assert any("pages/models.js" in p for p in problems)


def test_unstaged_deletion_of_precached_file_is_flagged(tmp_path, monkeypatch):
    """The same deletion left unstaged must flag too (kept as a distinct case so a
    regression cannot pass by handling only one of the two states)."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "models.js").unlink()

    problems = ch._sw_cache_bump_violations()

    assert any("pages/models.js" in p for p in problems)


def test_dropping_a_shell_entry_cannot_disarm_the_gate(tmp_path, monkeypatch):
    """The watch set must not be defined by the WORKING tree's SHELL, or a dev could
    silence the gate by deleting the entry for the file they just changed."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "models.js").write_text("// v2 change\n", encoding="utf-8")
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace(', "/pages/models.js"', ""),
                  encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert any("pages/models.js" in p and "CACHE" in p for p in problems), \
        "removing a file's SHELL entry must not silence its staleness"


def test_sw_js_itself_does_not_self_flag(tmp_path, monkeypatch):
    """sw.js gates the others and cannot gate itself: editing it (a comment, or the
    SHELL list) must not demand a CACHE bump for sw.js AS AN ASSET, or every bump commit
    would flag itself. Had no coverage; a mutant dropping the exclusion survived."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    sw = static / "sw.js"
    sw.write_text("// a comment-only edit, no CACHE change\n"
                  + sw.read_text(encoding="utf-8"), encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert not any("sw.js:" in p and "was not bumped" in p for p in problems), problems


def test_uncached_paths_are_not_watched(tmp_path, monkeypatch):
    """sw.js's fetch handler returns early for /api, /v1, /plugins and /localm-ca.crt,
    so those can never go stale from a missed bump and must not demand one. _SW_UNCACHED
    had NO coverage in either direction; mutants neutering it survived."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)

    assert ch._sw_is_cacheable(f"{_STATIC}/pages/models.js")
    assert ch._sw_is_cacheable(f"{_STATIC}/vendor/jsQR.min.js")
    assert not ch._sw_is_cacheable(f"{_STATIC}/api/thing.json")
    assert not ch._sw_is_cacheable(f"{_STATIC}/v1/thing.json")
    assert not ch._sw_is_cacheable(f"{_STATIC}/localm-ca.crt")
    assert not ch._sw_is_cacheable(_STATIC + "/sw.js"), "sw.js cannot gate itself"
    assert not ch._sw_is_cacheable("scripts/check_hygiene.py"), "outside the static tree"
    # The off-by-one that would break the ^-anchored regex: a leading "/" left on the
    # relative path makes _SW_UNCACHED never match again.
    assert not ch._sw_is_cacheable(f"{_STATIC}/api/nested/deep.json")


# ---- rule 5: a check that cannot run must say so, never pass silently -----------

def test_unparseable_cache_constant_fails_loud(tmp_path, monkeypatch):
    """A benign reformat (single quotes) must not silently disable the gate forever.
    Regex-based parsers rot; the rot must be loud."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace(
        'const CACHE = "localm-shell-v1";', "const CACHE = 'localm-shell-v1';"),
        encoding="utf-8")
    (static / "pages" / "settings.js").write_text("// v2\n", encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert problems, "an unparseable CACHE must fail loud, not silently pass"
    assert any("CACHE" in p and "sw.js" in p for p in problems)


def test_unparseable_shell_array_fails_loud(tmp_path, monkeypatch):
    """Same for SHELL: an empty parse means "I could not read it", never "nothing is
    precached"."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace("const SHELL = [",
                                                         "const SHELL=["),
                  encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert problems, "an unparseable SHELL array must fail loud, not silently pass"
    # Assert the PARSE diagnosis specifically. Asserting merely `"SHELL" in p` passed
    # even with the guard removed, because execution then fell through to the coverage
    # check, which says "... NOT listed in sw.js's SHELL" for every module - loud, but
    # the wrong diagnosis (it tells you to add N files rather than "I cannot read it").
    assert any("could not parse the SHELL" in p for p in problems), problems


def test_moved_sw_js_fails_loud(tmp_path, monkeypatch):
    """If sw.js moves, the gate is pointed at nothing. That must be loud - the static
    tree still shipping assets is the tell that this is a move, not a GUI-less checkout."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "sw.js").unlink()

    problems = ch._sw_cache_bump_violations()

    assert problems, "a missing sw.js beside a live static tree must fail loud"
    assert any("sw.js" in p for p in problems)


def test_absent_static_tree_is_silent(tmp_path, monkeypatch):
    """The genuinely benign case: no GUI service worker and no assets at all means
    there is nothing to gate. Distinguishing this from the move above is the point."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("no gui here\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "baseline")

    assert ch._sw_cache_bump_violations() == []


# ---- SHELL coverage: the precache list must match what ships --------------------

def test_shell_entry_with_no_file_behind_it_is_flagged(tmp_path, monkeypatch):
    """install() drops a missing precache entry silently (Promise.allSettled), so a
    typo'd SHELL path is invisible at runtime. Catch it here instead."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace(
        '"/pages/models.js"', '"/pages/modles.js"'), encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert any("modles.js" in p for p in problems), \
        "a SHELL entry naming a non-existent file must be flagged"


def test_shell_module_missing_from_precache_is_flagged(tmp_path, monkeypatch):
    """sw.js promises SHELL holds every app/* and pages/* module. A new page module
    that nobody adds to SHELL is not precached (so the PWA cannot open it offline).
    Enforce the promise rather than documenting it."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "brand_new.js").write_text("// forgot to add to SHELL\n",
                                                   encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "add page, forget SHELL")

    problems = ch._sw_cache_bump_violations()

    assert any("brand_new.js" in p and "SHELL" in p for p in problems), \
        "a shipped page module absent from SHELL must be flagged"


# ---- the gate must be real against the REAL sw.js and really wired in -----------

def test_real_sw_js_parses(monkeypatch):
    """Pins the REAL sw.js's format. Without this, a reformat of the production file
    would silently reduce the whole gate to a no-op and NO test would notice - the
    fixture above would keep passing forever."""
    ch = _load_check_hygiene()
    sw_text = (REPO_ROOT / ch._SW_JS).read_text(encoding="utf-8")

    cache = ch._sw_cache_version(sw_text)
    shell = ch._sw_shell_files(sw_text)

    assert cache and cache.startswith("localm-shell-v"), \
        f"the real sw.js's CACHE no longer parses (got {cache!r}) - the gate is dead"
    assert len(shell) > 20, \
        f"the real sw.js's SHELL no longer parses (got {len(shell)} entries) - gate is dead"


def test_real_repo_is_clean_under_the_gate():
    """The happy path on the real checkout: `python scripts/check_hygiene.py` must
    still pass. Guards against the coverage checks false-positiving on production."""
    ch = _load_check_hygiene()
    sw_text = (REPO_ROOT / ch._SW_JS).read_text(encoding="utf-8")
    shell = ch._sw_shell_files(sw_text)

    problems = ch._sw_shell_coverage_problems(shell, REPO_ROOT / ch._SW_STATIC)

    assert problems == [], f"the real sw.js SHELL is out of sync with the tree: {problems}"


def test_gate_is_wired_into_main(tmp_path, monkeypatch, capsys):
    """The check must actually be CALLED by main(), and its findings must make the
    process exit non-zero. A perfect check nobody runs is decoration; nothing else
    here would catch it being dropped from main()."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "settings.js").write_text("// v2 real change\n", encoding="utf-8")
    # Keep the run focused on check 6: neutralise the sibling gates, which have their
    # own tests and would otherwise need a whole synthetic repo to satisfy.
    monkeypatch.setattr(ch, "_scan", lambda f: [])
    monkeypatch.setattr(ch, "_changelog_append_only", lambda: [])
    monkeypatch.setattr(ch, "_raw_accessor_violations", lambda files: [])
    monkeypatch.setattr(ch, "_big_test_write_violations", lambda files: [])
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])

    rc = ch.main([])

    assert rc == 1, "main() must exit non-zero when a cached asset changed without a bump"
    assert "settings.js" in capsys.readouterr().err
