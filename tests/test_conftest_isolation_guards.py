# SPDX-License-Identifier: AGPL-3.0-or-later
"""The isolation guards in tests/conftest.py, run through the REAL conftest in a
sub-run: a test that leaves process-wide state changed must not reach the next
test on the same worker."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REPO = Path(__file__).resolve().parent.parent
_REAL_CONFTEST = str(Path(__file__).resolve().parent / "conftest.py")


def _with_real_conftest(pytester, *names):
    """Re-export the named conftest hooks and fixtures into the sub-run. A
    ``from conftest import *`` skips every underscore-prefixed name, so each is
    bound explicitly."""
    pytester.makeconftest(
        "import importlib.util as _u, sys\n"
        "_s = _u.spec_from_file_location('_localm_real_conftest', r'" + _REAL_CONFTEST + "')\n"
        "_m = _u.module_from_spec(_s)\n"
        "sys.modules['_localm_real_conftest'] = _m\n"
        "_s.loader.exec_module(_m)\n"
        + "".join(f"{name} = _m.{name}\n" for name in names))


@pytest.fixture
def subrun_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(_REPO))


def test_a_test_that_changes_tmpdir_fails_and_the_next_test_sees_the_original(
        pytester, subrun_env):
    _with_real_conftest(pytester, "pytest_runtest_logstart", "pytest_runtest_teardown")
    pytester.makepyfile(
        "import os\n"
        "START = os.environ.get('TMPDIR')\n"
        "def test_a_leaks():\n"
        "    os.environ['TMPDIR'] = os.path.join(os.getcwd(), 'gone')\n"
        "def test_b_sees_the_original():\n"
        "    assert os.environ.get('TMPDIR') == START\n")
    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider", "-p", "no:randomly")
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(["*left the process environment changed*TMPDIR*"])


def test_a_change_scoped_with_monkeypatch_is_not_a_leak(pytester, subrun_env):
    _with_real_conftest(pytester, "pytest_runtest_logstart", "pytest_runtest_teardown")
    pytester.makepyfile(
        "import os\n"
        "def test_scoped(monkeypatch):\n"
        "    monkeypatch.setenv('TMPDIR', os.path.join(os.getcwd(), 'scoped'))\n")
    pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider").assert_outcomes(passed=1)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="multiprocessing keeps the spawn executable as bytes only off Windows")
def test_the_mp_spawn_tests_leave_the_spawn_executable_as_they_found_it(pytester, subrun_env):
    pytester.makeconftest("")
    probe = pytester.makepyfile(
        test_zz_probe=(
            "import multiprocessing.spawn\n"
            "def test_the_executable_is_still_bytes():\n"
            "    assert isinstance(multiprocessing.spawn.get_executable(), bytes)\n"))
    spawn_fix = _REPO / "tests" / "test_mp_spawn_fix.py"
    result = pytester.runpytest_subprocess(
        str(spawn_fix), str(probe), "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
        "--rootdir", str(_REPO))
    assert result.ret == 0, result.stdout.str()
