# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/ci_runner_files.py: the single site under scripts/ that opens a
file named by a GitHub Actions runner variable (GITHUB_STEP_SUMMARY,
GITHUB_OUTPUT), and the sentinel that keeps it single.

The sentinel walks every scripts/*.py AST: any other read of those two
variable names is a second sink, which is what the shared helper exists to
prevent. A script that wants to publish imports the helper instead.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
HELPER = SCRIPTS / "ci_runner_files.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


crf = _load(HELPER)


class TestAppend:
    def test_appends_to_the_named_file_and_reports_true(self, tmp_path, monkeypatch):
        target = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        assert crf.append(crf.STEP_SUMMARY, "one\n") is True
        assert crf.append(crf.STEP_SUMMARY, "two\n") is True
        assert target.read_text(encoding="utf-8") == "one\ntwo\n"

    def test_github_output_is_a_separate_file(self, tmp_path, monkeypatch):
        summary, output = tmp_path / "s.md", tmp_path / "o.txt"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        assert crf.append(crf.OUTPUT, "touched=true\n") is True
        assert output.read_text(encoding="utf-8") == "touched=true\n"
        assert not summary.exists()

    @pytest.mark.parametrize("value", [None, ""])
    def test_unset_or_empty_variable_writes_nothing_and_reports_false(self, tmp_path, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        else:
            monkeypatch.setenv("GITHUB_STEP_SUMMARY", value)
        assert crf.append(crf.STEP_SUMMARY, "x") is False
        assert list(tmp_path.iterdir()) == []

    def test_unknown_variable_is_refused(self, monkeypatch):
        monkeypatch.setenv("HOME_OF_EVIL", "anything")
        with pytest.raises(ValueError):
            crf.append("HOME_OF_EVIL", "x")

    def test_unwritable_target_raises_oserror(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "missing-dir" / "s.md"))
        with pytest.raises(OSError):
            crf.append(crf.STEP_SUMMARY, "x")


def _env_name_reads(tree: ast.AST) -> set[str]:
    """Every string literal naming a runner file variable that reaches an
    ``os.environ`` / ``os.getenv`` read in *tree*."""
    hits: set[str] = set()
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Subscript):
            target = node.slice
        elif isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name in ("get", "getenv", "pop", "setdefault") and node.args:
                target = node.args[0]
        if isinstance(target, ast.Constant) and target.value in crf.RUNNER_FILE_VARS:
            src = ast.get_source_segment(_SOURCES[tree], node) or ""
            if "environ" in src or "getenv" in src:
                hits.add(target.value)
    return hits


_SOURCES: dict[ast.AST, str] = {}


class TestSingleSink:
    def test_no_script_reads_a_runner_file_variable_except_the_helper(self):
        offenders = []
        for path in sorted(SCRIPTS.glob("*.py")):
            if path == HELPER:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            _SOURCES[tree] = source
            hits = _env_name_reads(tree)
            if hits:
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {sorted(hits)}")
        assert not offenders, (
            "runner file variables read outside scripts/ci_runner_files.py - "
            "publish through ci_runner_files.append() instead, so the runner-owned "
            "file is opened at exactly one site:\n" + "\n".join(offenders))

    def test_the_helper_is_the_one_site_and_is_marked(self):
        source = HELPER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        _SOURCES[tree] = source
        assert _env_name_reads(tree) == set(), "the helper reads var, never a literal name"
        opens = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "open"]
        assert len(opens) == 1
        line_before = source.splitlines()[opens[0].lineno - 2].strip()
        assert line_before.startswith("# codeql[py/path-injection]"), line_before

    def test_every_publishing_script_imports_the_helper(self):
        publishers = [p for p in SCRIPTS.glob("*.py")
                      if p != HELPER and "ci_runner_files.append(" in p.read_text(encoding="utf-8")]
        assert len(publishers) >= 8, [p.name for p in publishers]
        for p in publishers:
            assert "import ci_runner_files" in p.read_text(encoding="utf-8"), p.name
