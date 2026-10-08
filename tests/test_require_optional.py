# SPDX-License-Identifier: AGPL-3.0-or-later
"""LOCALM_REQUIRE_OPTIONAL turns a skip for a missing optional package or
resource into a failure, and leaves every other skip alone.

The end-to-end cases run a real pytest in a fresh directory that registers the
same hooks, so they exercise the hook plumbing (collection and test reports),
not just the matching functions.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests import _require_optional as ro

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestMatching:
    def test_an_importorskip_reason_names_the_package(self):
        reason = "could not import 'torch': No module named 'torch'"
        assert ro.skip_subject(reason) == "torch"

    def test_a_dotted_module_is_matched_by_its_top_level_name(self):
        reason = "could not import 'PIL.Image': No module named 'PIL'"
        assert ro.forbidden_subject(reason, frozenset({"PIL"})) == "PIL.Image"

    def test_a_resource_gate_reason_names_the_marker(self):
        reason = "real_gguf: native llama runtime not provisioned (run 'localm setup-llama')"
        assert ro.skip_subject(reason) == "real_gguf"

    @pytest.mark.parametrize("reason", [
        "GetDriveTypeW is Windows-only",
        "POSIX modes only",
        "no display",
        "CUDA/ROCm not available",
        "git not on PATH",
    ])
    def test_platform_and_hardware_skips_are_never_forbidden(self, reason):
        assert ro.skip_subject(reason) is None
        assert ro.forbidden_subject(reason, frozenset({"*"})) is None

    def test_only_listed_subjects_are_forbidden(self):
        reason = "could not import 'torch': No module named 'torch'"
        assert ro.forbidden_subject(reason, frozenset({"psutil"})) is None
        assert ro.forbidden_subject(reason, frozenset({"psutil", "torch"})) == "torch"

    def test_a_star_forbids_every_missing_package_and_gate(self):
        assert ro.forbidden_subject("could not import 'x'", frozenset({"*"})) == "x"
        assert ro.forbidden_subject("real_comfy: set LOCALM_TEST_COMFY_URL",
                                    frozenset({"*"})) == "real_comfy"

    def test_nothing_is_forbidden_when_nothing_is_required(self):
        assert ro.forbidden_subject("could not import 'torch'", frozenset()) is None

    def test_the_environment_variable_is_parsed(self, monkeypatch):
        monkeypatch.setenv(ro.REQUIRE_ENV, " torch, psutil ,,real_gguf ")
        assert ro.required_names() == frozenset({"torch", "psutil", "real_gguf"})
        monkeypatch.delenv(ro.REQUIRE_ENV)
        assert ro.required_names() == frozenset()


_CONFTEST = """
from tests._require_optional import (
    pytest_make_collect_report,
    pytest_runtest_makereport,
)
"""

_TESTS = """
import pytest


def test_in_test_importorskip():
    pytest.importorskip("localm_no_such_package_a")


def test_unrelated_skip():
    pytest.skip("Windows only")


def test_resource_gate_skip():
    pytest.skip("real_gguf: native llama runtime not provisioned")


def test_plain_pass():
    assert True
"""

_MODULE_LEVEL = """
import pytest

pytest.importorskip("localm_no_such_package_b")


def test_never_runs():
    assert True
"""


def _run(tmp_path: Path, require: str | None) -> dict:
    (tmp_path / "conftest.py").write_text(textwrap.dedent(_CONFTEST), encoding="utf-8")
    (tmp_path / "test_cases.py").write_text(textwrap.dedent(_TESTS), encoding="utf-8")
    (tmp_path / "test_module_level.py").write_text(textwrap.dedent(_MODULE_LEVEL),
                                                   encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != ro.REQUIRE_ENV}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if require is not None:
        env[ro.REQUIRE_ENV] = require
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly",
         "--rootdir", str(tmp_path), "--continue-on-collection-errors", "-q", "-rfEs",
         str(tmp_path)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
    return {"code": proc.returncode, "out": proc.stdout + proc.stderr}


def test_with_nothing_required_every_missing_package_is_a_skip(tmp_path):
    result = _run(tmp_path, None)
    assert result["code"] == 0, result["out"]
    assert "1 passed" in result["out"]
    assert "3 skipped" in result["out"] or "4 skipped" in result["out"]


def test_a_required_package_fails_both_in_a_test_and_at_collection(tmp_path):
    result = _run(tmp_path, "localm_no_such_package_a,localm_no_such_package_b")
    out = result["out"]
    assert result["code"] != 0, out
    assert "LOCALM_REQUIRE_OPTIONAL requires 'localm_no_such_package_a'" in out
    assert "LOCALM_REQUIRE_OPTIONAL requires 'localm_no_such_package_b'" in out


def test_a_required_resource_gate_fails(tmp_path):
    result = _run(tmp_path, "real_gguf")
    assert result["code"] != 0, result["out"]
    assert "requires 'real_gguf'" in result["out"]


def test_a_skip_for_another_reason_is_left_alone_even_under_star(tmp_path):
    result = _run(tmp_path, "*")
    out = result["out"]
    assert "requires 'localm_no_such_package_a'" in out
    assert "SKIPPED [1] test_cases.py" in out and "Windows only" in out
    assert "FAILED test_cases.py::test_unrelated_skip" not in out
    assert "1 passed" in out


def test_a_package_that_is_not_listed_still_skips(tmp_path):
    result = _run(tmp_path, "some_other_package")
    assert result["code"] == 0, result["out"]
