# SPDX-License-Identifier: AGPL-3.0-or-later
"""tests/conftest.py puts the GPU-visibility variables back after every test, so
a test that sets one directly on os.environ does not hide the GPU from the tests
after it in the same worker."""

from __future__ import annotations

from pathlib import Path

pytest_plugins = ["pytester"]

_REAL_CONFTEST = str(Path(__file__).parent / "conftest.py").replace("\\", "/")
_REPO_ROOT = str(Path(__file__).resolve().parent.parent).replace("\\", "/")


def test_a_variable_a_test_sets_is_gone_in_the_next_test(pytester, monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    pytester.makeconftest(
        "import importlib.util as _u, sys\n"
        "sys.path.insert(0, r'" + _REPO_ROOT + "')\n"
        "_s = _u.spec_from_file_location('_localm_real_conftest', r'" + _REAL_CONFTEST + "')\n"
        "_m = _u.module_from_spec(_s)\n"
        "sys.modules['_localm_real_conftest'] = _m\n"
        "_s.loader.exec_module(_m)\n"
        "_restore_gpu_visibility_env = _m._restore_gpu_visibility_env\n")
    pytester.makepyfile(
        "import os\n\n\n"
        "def test_1_sets_them():\n"
        "    os.environ['HIP_VISIBLE_DEVICES'] = '-1'\n"
        "    os.environ['ROCR_VISIBLE_DEVICES'] = '-1'\n"
        "    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'\n\n\n"
        "def test_2_sees_them_as_before():\n"
        "    assert 'HIP_VISIBLE_DEVICES' not in os.environ\n"
        "    assert 'ROCR_VISIBLE_DEVICES' not in os.environ\n"
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'\n")
    result = pytester.runpytest_subprocess("-p", "no:randomly", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=2)
