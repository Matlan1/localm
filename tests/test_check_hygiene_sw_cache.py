# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py's PWA service-worker cache-derivation check (check 6).

sw.js's CACHE constant is not a hand-typed version string in git:
localm/plugins/gui/web.py's GET /sw.js route computes it fresh on every request
from a content digest of the static assets being served. This file covers what
check_hygiene.py enforces on the tracked file: SHELL precache coverage, and that
the CACHE placeholder line is still in the shape web.py's route expects to
substitute into.

Every "clean" assertion here is paired with a positive control that must FAIL,
and each silent-skip path (unparseable CACHE, unparseable SHELL, a moved
sw.js) is asserted to be LOUD rather than merely absent.
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


# Mirrors the real sw.js's shape: a CACHE placeholder and a SHELL precache list.
_SW_JS_TEXT = (
    'const CACHE = "localm-shell-dev";\n'
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
    (static / "sw.js").write_text(_SW_JS_TEXT, encoding="utf-8")
    (static / "index.html").write_text("<html></html>", encoding="utf-8")
    (static / "style.css").write_text("body {}", encoding="utf-8")
    (static / "app" / "main.js").write_text("// main v1\n", encoding="utf-8")
    (static / "pages" / "settings.js").write_text("// settings v1\n", encoding="utf-8")
    (static / "pages" / "models.js").write_text("// models v1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("not a static asset\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "baseline")
    return static


# ---- the core contract: SHELL coverage + a parseable placeholder ----------------

def test_no_change_is_clean(tmp_path, monkeypatch):
    """Happy path: an untouched checkout passes. Paired with the positive controls
    below, which prove this is a real [] and not a hardwired one."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_fake_repo(tmp_path)
    assert ch._sw_cache_derivation_violations() == []


def test_asset_content_changing_alone_is_not_flagged(tmp_path, monkeypatch):
    """CACHE is computed at request time in web.py, so a cached asset's bytes
    changing is not this check's business; only SHELL coverage and the
    placeholder shape are."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "settings.js").write_text("// v2 real change\n", encoding="utf-8")

    assert ch._sw_cache_derivation_violations() == []


# ---- a check that cannot run says so, never passes silently -------------------

def test_unparseable_cache_constant_fails_loud(tmp_path, monkeypatch):
    """A reformat (single quotes) must not silently disable this check, and must
    be flagged for the RIGHT reason: web.py's route would fail to find the line to
    substitute into, so every client would get a 500 instead of a service
    worker."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace(
        'const CACHE = "localm-shell-dev";', "const CACHE = 'localm-shell-dev';"),
        encoding="utf-8")

    problems = ch._sw_cache_derivation_violations()

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

    problems = ch._sw_cache_derivation_violations()

    assert problems, "an unparseable SHELL array must fail loud, not silently pass"
    # Assert the PARSE diagnosis specifically: a bare SHELL-in-output check also
    # matches the coverage failure, which is a different diagnosis.
    assert any("could not parse the SHELL" in p for p in problems), problems


def test_moved_sw_js_fails_loud(tmp_path, monkeypatch):
    """If sw.js moves, the check (and web.py's route) are pointed at nothing. That
    must be loud - the static tree still shipping assets is the tell that this is a
    move, not a GUI-less checkout."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "sw.js").unlink()

    problems = ch._sw_cache_derivation_violations()

    assert problems, "a missing sw.js beside a live static tree must fail loud"
    assert any("sw.js" in p for p in problems)


def test_absent_static_tree_is_silent(tmp_path, monkeypatch):
    """The genuinely benign case: no GUI service worker and no assets at all means
    there is nothing to check. Distinguishing this from the move above is the point."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("no gui here\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "baseline")

    assert ch._sw_cache_derivation_violations() == []


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

    problems = ch._sw_cache_derivation_violations()

    assert any("modles.js" in p for p in problems), \
        "a SHELL entry naming a non-existent file must be flagged"


def test_shell_module_missing_from_precache_is_flagged(tmp_path, monkeypatch):
    """sw.js promises SHELL holds every app/* and pages/* module. A new page module
    that nobody adds to SHELL is not precached, so the PWA cannot open it
    offline."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    static = _init_fake_repo(tmp_path)
    (static / "pages" / "brand_new.js").write_text("// forgot to add to SHELL\n",
                                                   encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "add page, forget SHELL")

    problems = ch._sw_cache_derivation_violations()

    assert any("brand_new.js" in p and "SHELL" in p for p in problems), \
        "a shipped page module absent from SHELL must be flagged"


# ---- the gate must be real against the REAL sw.js and really wired in -----------

def test_real_sw_js_parses(monkeypatch):
    """Pins the REAL sw.js's format. Without this, a reformat of the production file
    would silently reduce this check to a no-op and NO test would notice - the
    fixture above would keep passing forever."""
    ch = _load_check_hygiene()
    sw_text = (REPO_ROOT / ch._SW_JS).read_text(encoding="utf-8")

    cache = ch._sw_cache_version(sw_text)
    shell = ch._sw_shell_files(sw_text)

    assert cache and cache.startswith("localm-shell-"), \
        f"the real sw.js's CACHE no longer parses (got {cache!r}) - the check is dead"
    assert len(shell) > 20, \
        f"the real sw.js's SHELL no longer parses (got {len(shell)} entries) - dead"


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
    sw = static / "sw.js"
    sw.write_text(sw.read_text(encoding="utf-8").replace(
        '"/pages/models.js"', '"/pages/modles.js"'), encoding="utf-8")
    # Neutralise the sibling gates so the run exercises check 6 alone.
    monkeypatch.setattr(ch, "_scan", lambda f: [])
    monkeypatch.setattr(ch, "_changelog_append_only", lambda: [])
    monkeypatch.setattr(ch, "_raw_accessor_violations", lambda files: [])
    monkeypatch.setattr(ch, "_big_test_write_violations", lambda files: [])
    monkeypatch.setattr(ch, "_never_tracked_violations", lambda: [])

    rc = ch.main([])

    assert rc == 1, "main() must exit non-zero on a SHELL-coverage violation"
    assert "modles.js" in capsys.readouterr().err
