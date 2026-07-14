# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/check_hygiene.py's PWA service-worker cache-version bump gate
(check 6). Shipped twice undetected by a human review before this check
existed (v49 for #621's settings.js fix; again for the managed_comfy_enabled
checkbox removal) - see check_hygiene.py's own comment above
_sw_cache_bump_violations for the incident record.
"""

import importlib.util
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_hygiene():
    spec = importlib.util.spec_from_file_location(
        "check_hygiene", REPO_ROOT / "scripts" / "check_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SW_JS_TEXT = (
    'const CACHE = "localm-shell-v1";\n'
    'const SHELL = [\n'
    '  "/", "/index.html", "/style.css",\n'
    '  "/pages/settings.js", "/pages/models.js",\n'
    '];\n'
)


def _init_fake_repo(tmp_path: Path) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, env=env)
    static = tmp_path / "localm" / "plugins" / "gui" / "static"
    (static / "pages").mkdir(parents=True)
    (static / "sw.js").write_text(_SW_JS_TEXT, encoding="utf-8")
    (static / "index.html").write_text("<html></html>", encoding="utf-8")
    (static / "style.css").write_text("body {}", encoding="utf-8")
    (static / "pages" / "settings.js").write_text("// settings v1\n", encoding="utf-8")
    (static / "pages" / "models.js").write_text("// models v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True, env=env)


def test_no_change_is_clean(tmp_path, monkeypatch):
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_fake_repo(tmp_path)
    assert ch._sw_cache_bump_violations() == []


def test_precached_file_changed_without_cache_bump_fails(tmp_path, monkeypatch):
    """The exact regression this check exists for: a precached file changes,
    CACHE does not - must be flagged, naming the changed file."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_fake_repo(tmp_path)
    (tmp_path / "localm/plugins/gui/static/pages/settings.js").write_text(
        "// settings v2 - a real behavior change\n", encoding="utf-8")

    problems = ch._sw_cache_bump_violations()

    assert problems, "changing a precached file without bumping CACHE must be flagged"
    assert any("pages/settings.js" in p and "CACHE" in p for p in problems)


def test_precached_file_changed_with_cache_bump_is_clean(tmp_path, monkeypatch):
    """The correct fix: bumping CACHE alongside the file change clears the flag."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_fake_repo(tmp_path)
    sw = tmp_path / "localm/plugins/gui/static/sw.js"
    sw.write_text(_SW_JS_TEXT.replace("v1", "v2"), encoding="utf-8")
    (tmp_path / "localm/plugins/gui/static/pages/settings.js").write_text(
        "// settings v2 - a real behavior change\n", encoding="utf-8")

    assert ch._sw_cache_bump_violations() == []


def test_non_precached_file_changing_is_not_flagged(tmp_path, monkeypatch):
    """A file NOT listed in SHELL (models.js is listed; add an unlisted one)
    changing must not be flagged - only precached assets matter."""
    ch = _load_check_hygiene()
    monkeypatch.setattr(ch, "REPO", tmp_path)
    _init_fake_repo(tmp_path)
    unlisted = tmp_path / "localm/plugins/gui/static/pages/not_precached.js"
    unlisted.write_text("// never in SHELL\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "add unlisted file"], cwd=tmp_path, check=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})
    unlisted.write_text("// changed, but not precached\n", encoding="utf-8")

    assert ch._sw_cache_bump_violations() == []
