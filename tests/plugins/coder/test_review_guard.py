# SPDX-License-Identifier: AGPL-3.0-or-later
"""Review guard: classify the changed files a passing check cannot vouch for
(rewritten tests / weakened CI config) so the CLI can surface them for review.
"""

from __future__ import annotations

from localm.plugins.coder.review_guard import (
    classify_sensitive_changes,
    render_warning,
)


def _cf(*paths):
    return [{"path": p} for p in paths]


def test_classifies_python_test_files():
    flags = classify_sensitive_changes(_cf(
        "tests/test_foo.py", "src/app.py", "foo_test.py", "pkg/conftest.py"))
    assert set(flags["tests"]) == {"tests/test_foo.py", "foo_test.py",
                                   "pkg/conftest.py"}
    assert "src/app.py" not in flags["tests"]


def test_classifies_js_ts_test_files():
    flags = classify_sensitive_changes(_cf(
        "ui/Button.test.tsx", "ui/util.spec.js", "ui/util.ts"))
    assert set(flags["tests"]) == {"ui/Button.test.tsx", "ui/util.spec.js"}


def test_classifies_ci_config():
    flags = classify_sensitive_changes(_cf(
        ".github/workflows/ci.yml", ".pre-commit-config.yaml", "ruff.toml",
        ".flake8", "README.md"))
    assert set(flags["ci_config"]) == {".github/workflows/ci.yml",
                                       ".pre-commit-config.yaml", "ruff.toml",
                                       ".flake8"}
    assert "README.md" not in flags["ci_config"]
    assert "README.md" not in flags["tests"]


def test_plain_source_is_not_flagged():
    flags = classify_sensitive_changes(_cf(
        "src/main.py", "lib/helpers.py", "docs/guide.md", "pyproject.toml"))
    assert flags == {"tests": [], "ci_config": []}     # incl. pyproject


def test_windows_paths_are_normalised():
    flags = classify_sensitive_changes([
        {"path": "tests\\test_x.py"},
        {"path": ".github\\workflows\\ci.yml"},
    ])
    assert "tests/test_x.py" in flags["tests"]
    assert ".github/workflows/ci.yml" in flags["ci_config"]


def test_a_file_is_at_most_one_category():
    # conftest under tests/ is a test, never double-counted as config.
    flags = classify_sensitive_changes(_cf("tests/conftest.py"))
    assert flags["tests"] == ["tests/conftest.py"]
    assert flags["ci_config"] == []


def test_accepts_plain_path_strings_and_dedupes():
    flags = classify_sensitive_changes(["tests/test_a.py", "tests/test_a.py"])
    assert flags["tests"] == ["tests/test_a.py"]


def test_render_warning_empty_when_nothing_sensitive():
    assert render_warning({"tests": [], "ci_config": []}) == ""
    assert render_warning(classify_sensitive_changes(_cf("src/app.py"))) == ""


def test_render_warning_lists_files_and_says_review():
    msg = render_warning({"tests": ["tests/test_x.py"], "ci_config": [".flake8"]})
    assert "tests/test_x.py" in msg
    assert ".flake8" in msg
    assert "review" in msg.lower()


def test_cli_warn_helper_surfaces_edits(capsys):
    import localm.plugins.coder.cli as cli

    class _Agent:
        def changed_files(self):
            return [{"path": "tests/test_x.py"}, {"path": "src/ok.py"}]

    cli._warn_sensitive_changes(_Agent())
    captured = capsys.readouterr()
    assert "tests/test_x.py" in (captured.out + captured.err)


def test_cli_warn_helper_silent_when_nothing_sensitive(capsys):
    import localm.plugins.coder.cli as cli

    class _Agent:
        def changed_files(self):
            return [{"path": "src/app.py"}]

    cli._warn_sensitive_changes(_Agent())
    captured = capsys.readouterr()
    assert "review" not in (captured.out + captured.err).lower()


def test_cli_warn_helper_never_raises():
    import localm.plugins.coder.cli as cli

    class _Boom:
        def changed_files(self):
            raise RuntimeError("boom")

    cli._warn_sensitive_changes(_Boom())     # best-effort: must not propagate
