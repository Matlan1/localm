# SPDX-License-Identifier: AGPL-3.0-or-later
"""The import-time LOCALM_HOME must be cleaned up, and the cleanup must not go
silently blind.

conftest.py creates its throwaway ``LOCALM_HOME`` with ``mkdtemp`` at IMPORT
time, so that any module importing ``localm.config`` during collection already
resolves ``HOME_DIR`` to a temp dir. Import time means once per PROCESS, which
includes processes that never finish a pytest session: test_conftest_disk_guard
and test_conftest_syspath_guard both exec the real conftest by path, and so does
any script written to unit-check one of those guards. With cleanup only in
``pytest_sessionfinish``, every one of those leaks a directory permanently, and
they accumulate in hundreds.

The naive fix breaks something. That per-process directory is what a run uses to
answer "did the TMPDIR/TEMP/TMP redirect reach every xdist worker?": a worker
that missed it necessarily left its directory on the wrong root, so counting
leftovers there is the check. Removing every directory on exit makes that count
read zero either way - always-green, the same failure shape as a guard that
never arms. So the cleanup is conditional, and this file pins both halves:

  (i)  a process that lands on the intended temp root cleans up after itself;
  (ii) a process that does NOT keeps its directory AND says so, loudly.

Every claim here is fires-controlled against its own inverse in the same run:
half (i) is paired with a run that unregisters the handler and must then see the
directory survive, and half (ii) with a matching run that must stay green. A
cleanup nobody proved can fail, and a detector nobody proved can fire, are both
worth nothing.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REAL_CONFTEST = str(Path(__file__).parent / "conftest.py").replace("\\", "/")

# Loads the SHIPPED tests/conftest.py by path under its own module name, reports
# the directory it made, and applies one variation.
_PROBE = r'''
import atexit, importlib.util, os, sys

conftest_path, mode, stamped_root = sys.argv[1], sys.argv[2], sys.argv[3]

spec = importlib.util.spec_from_file_location("_localm_cleanup_probe", conftest_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

home = module._test_home_dir
# Anti-vacuity: without this, "gone afterwards" would also pass for a directory
# that was never created, and the whole file would be theater.
assert os.path.isdir(home), "the import-time LOCALM_HOME was never created"
assert os.environ["LOCALM_HOME"] == home, "LOCALM_HOME was not pinned to it"
print("CREATED=" + home)

class _Config:
    """The only thing pytest_configure reads off a config here."""

if mode == "unregister":
    # The fires-control for the cleanup: take the handler away and nothing else.
    assert atexit.unregister(module._cleanup_test_home) is None
    module._cleanup_test_home = None          # so nothing can re-register it
elif mode == "configure":
    module.pytest_configure(_Config())
elif mode == "configure-worker":
    config = _Config()
    config.workerinput = ({} if stamped_root == "-"
                          else {module._EXPECTED_TEMP_ROOT_KEY: stamped_root})
    module.pytest_configure(config)
elif mode == "twice":
    module._cleanup_test_home()
    module._cleanup_test_home()               # must be a no-op, not a raise
    assert not os.path.exists(home), "the first call did not remove it"
elif mode == "stamp":
    node = _Config()
    node.workerinput = {}
    module.pytest_configure_node(node)
    stamped = node.workerinput[module._EXPECTED_TEMP_ROOT_KEY]
    assert module._same_dir(stamped, os.path.dirname(home)), (
        "the controller stamped " + str(stamped) + ", not its own temp root")
    print("STAMPED=" + stamped)
elif mode != "plain":
    raise SystemExit("unknown mode " + mode)

if mode.startswith("configure"):
    print("EXPECTED=" + str(module._expected_temp_root))
    print("COMPLAINT=" + ("yes" if module._wrong_temp_root else "no"))
'''


def _run_probe(tmp_path, mode, *, temp_root=None, expected_env=None,
               stamped_root="-"):
    """Load the real conftest in a SUBPROCESS and return (home_dir, stdout).

    A subprocess, not an in-process import, because the thing under test is what
    happens when the PROCESS ends - and because the import has global side
    effects (it repoints LOCALM_HOME and arms the system-path guard) that have no
    business landing in this session.

    ``temp_root`` defaults to a directory inside the caller's own tmp_path, so a
    probe that is SUPPOSED to leave its directory behind leaves it somewhere
    disposable instead of in the real temp dir this file exists to stop
    littering."""
    root = Path(temp_root) if temp_root else (tmp_path / "temproot")
    root.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")

    env = dict(os.environ)
    # All three: tempfile's candidate list is TMPDIR, then TEMP, then TMP.
    env.update(TMPDIR=str(root), TEMP=str(root), TMP=str(root))
    env.pop("LOCALM_TEST_EXPECTED_TEMP_ROOT", None)
    if expected_env is not None:
        env["LOCALM_TEST_EXPECTED_TEMP_ROOT"] = str(expected_env)

    done = subprocess.run(
        [sys.executable, str(script), _REAL_CONFTEST, mode, str(stamped_root)],
        capture_output=True, text=True, env=env, timeout=180)
    assert done.returncode == 0, (
        f"the probe process failed:\n{done.stdout}\n{done.stderr}")
    created = [line.split("=", 1)[1] for line in done.stdout.splitlines()
               if line.startswith("CREATED=")]
    assert len(created) == 1, f"probe did not report its home dir:\n{done.stdout}"
    return created[0], done.stdout


class TestTheDirectoryIsCleanedUpWhenTheProcessExits:
    """Half (i), and the actual defect: a process that merely IMPORTS the
    conftest must not leave a directory behind."""

    def test_the_import_time_home_is_gone_after_the_process_exits(self, tmp_path):
        home, out = _run_probe(tmp_path, "plain")
        assert "CREATED=" in out
        assert not os.path.exists(home), (
            f"{home} survived a process that only imported tests/conftest.py - "
            "this is the leak: 426 of these had piled up in a real temp dir")

    def test_WITHOUT_the_handler_the_directory_survives(self, tmp_path):
        """The fires-control for the test above.

        Identical probe, one difference: the atexit handler is unregistered. If
        this ALSO came back clean, the cleanup would be proving nothing and
        something else (the OS, the harness, a temp sweeper) would be doing the
        removing."""
        home, _ = _run_probe(tmp_path, "unregister")
        assert os.path.exists(home), (
            "the directory vanished even with the cleanup handler removed, so "
            "the passing test above does not prove the handler does anything")
        # Ours, named explicitly, and inside this test's own tmp_path.
        os.rmdir(home)

    def test_the_temp_root_is_left_empty(self, tmp_path):
        """Nothing else lingers either: not a stray marker file, not a partial
        tree. The count a run makes is of everything matching the prefix."""
        root = tmp_path / "temproot"
        _run_probe(tmp_path, "plain", temp_root=root)
        assert glob.glob(str(root / "localm_test_home_*")) == []

    def test_cleaning_up_twice_is_harmless(self, tmp_path):
        """pytest_sessionfinish cleans up at the end of a session and atexit
        cleans up again at exit, so the second call always meets a directory that
        is already gone. It must not raise there - an exception inside an atexit
        handler prints a traceback and makes a green run look broken. The probe
        asserts the removal happened and then exits normally, so a raise in
        either the second call or the handler itself is a non-zero exit."""
        home, _ = _run_probe(tmp_path, "twice")
        assert not os.path.exists(home)


class TestAWrongTempRootIsDetectedAndItsEvidenceKept:
    """Half (ii): the replacement oracle. A process whose temp root is not the
    one the run intended must keep its directory (so counting leftovers still
    answers the old question) and must complain (so nobody has to go counting)."""

    def test_a_mismatch_keeps_the_directory(self, tmp_path):
        home, out = _run_probe(tmp_path, "configure",
                               expected_env=tmp_path / "somewhere-else")
        assert "COMPLAINT=yes" in out, "the mismatch was not even noticed"
        assert os.path.exists(home), (
            "the evidence was deleted: a leftover count would now read zero for "
            "a process that used the WRONG temp root, which is the always-green "
            "failure this design exists to avoid")
        os.rmdir(home)

    def test_a_MATCHING_root_is_cleaned_up_and_says_nothing(self, tmp_path):
        """The fires-control for the test above, and the no-crying-wolf half.

        Same code path, same env var, only the value differs: pointed at the root
        the process really used, it must neither complain nor leave anything. A
        detector that fired on the healthy case would be silenced within a week."""
        root = tmp_path / "temproot"
        home, out = _run_probe(tmp_path, "configure", temp_root=root,
                               expected_env=root)
        assert "COMPLAINT=no" in out, f"false positive on a matching root:\n{out}"
        assert not os.path.exists(home)

    def test_a_worker_is_judged_against_what_the_controller_stamped(self, tmp_path):
        """The worker path: the expectation arrives in workerinput, over
        execnet's channel, and is what the worker is measured against."""
        root = tmp_path / "temproot"
        home, out = _run_probe(tmp_path, "configure-worker", temp_root=root,
                               stamped_root=tmp_path / "the-other-drive")
        assert "COMPLAINT=yes" in out
        assert os.path.exists(home), "a misplaced worker must keep its evidence"
        os.rmdir(home)

    def test_a_stamped_root_beats_the_env_seam(self, tmp_path):
        """The env var can only ADD a check to a run that had none. If it could
        override what the controller stamped, anything that sets it (including
        this file's own fires-controls) could switch the real check off."""
        root = tmp_path / "temproot"
        home, out = _run_probe(tmp_path, "configure-worker", temp_root=root,
                               stamped_root=tmp_path / "the-other-drive",
                               expected_env=root)          # env says "all fine"
        assert "COMPLAINT=yes" in out, (
            "the env seam overrode the controller's stamp, so it can be used to "
            "disable the guard rather than only to add one")
        os.rmdir(home)

    def test_an_UNSTAMPED_worker_reports_that_the_check_is_not_armed(self, tmp_path):
        """The one way this oracle can silently fail to exist: a worker whose
        controller never loaded tests/conftest.py, so nothing stamped the
        expectation. Cleaning up quietly there would be exactly the always-green
        outcome, so it is reported instead."""
        root = tmp_path / "temproot"
        home, out = _run_probe(tmp_path, "configure-worker", temp_root=root,
                               stamped_root="-")           # workerinput, no key
        assert "COMPLAINT=yes" in out
        assert "NOT ARMED" in out or os.path.exists(home)
        assert os.path.exists(home)
        os.rmdir(home)

    def test_nothing_is_checked_when_no_root_was_declared(self, tmp_path):
        """An ordinary single-process ``pytest`` declares no intent and must not
        be nagged: no expectation, no complaint, and still no leak."""
        home, out = _run_probe(tmp_path, "configure")
        assert "EXPECTED=None" in out
        assert "COMPLAINT=no" in out
        assert not os.path.exists(home)


def _with_real_conftest(pytester, *, node_hook=False):
    """Point a sub-run at the REAL tests/conftest.py.

    By path under a distinct module name, and re-exporting only the names this
    file needs: ``from conftest import *`` silently skips every underscore-
    prefixed name, so the autouse fixture would never arrive and the sub-run
    would pass with nothing installed - green while proving nothing."""
    exports = (
        "pytest_configure = _m.pytest_configure\n"
        "pytest_sessionfinish = _m.pytest_sessionfinish\n"
        "_temp_root_is_the_one_the_run_intended = "
        "_m._temp_root_is_the_one_the_run_intended\n")
    if node_hook:
        exports += "pytest_configure_node = _m.pytest_configure_node\n"
    pytester.makeconftest(
        "import importlib.util as _u, sys\n"
        "_s = _u.spec_from_file_location('_localm_real_conftest', r'"
        + _REAL_CONFTEST + "')\n"
        "_m = _u.module_from_spec(_s)\n"
        "sys.modules['_localm_real_conftest'] = _m\n"
        "_s.loader.exec_module(_m)\n"
        + exports)


@pytest.fixture
def sub_temp_root(tmp_path, monkeypatch):
    """A disposable temp root for a sub-pytest run.

    pytester points the sub-run's own tmp dirs at PYTEST_DEBUG_TEMPROOT and never
    touches TMPDIR/TEMP/TMP, so this directory ends up holding the
    ``localm_test_home_*`` entries and nothing else - which is what makes
    "count what is left in here" a clean assertion.

    The expected-root env var is cleared, not just left alone: a sub-run inherits
    this process's environment, so a run that legitimately declares one (a fleet
    lane redirecting its temp root does exactly that) would otherwise hand the
    sub-run an expectation naming a root the sub-run was never going to use, and
    every test here would go red for a reason that has nothing to do with the
    code. Tests that need a value set one after this."""
    root = tmp_path / "subtemproot"
    root.mkdir()
    monkeypatch.delenv("LOCALM_TEST_EXPECTED_TEMP_ROOT", raising=False)
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, str(root))
    return root


class TestARealRunGoesRedAndKeepsTheEvidence:
    """The end-to-end half: a real pytest process, through the shipped hooks and
    the shipped fixture, not a hand-built config object."""

    def test_a_misplaced_run_fails_and_names_the_directory(
            self, pytester, sub_temp_root, monkeypatch):
        monkeypatch.setenv("LOCALM_TEST_EXPECTED_TEMP_ROOT",
                           str(sub_temp_root / "not-where-we-landed"))
        _with_real_conftest(pytester)
        pytester.makepyfile("def test_ordinary():\n    assert True\n")
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        assert result.ret != 0, (
            "a run on the wrong temp root exited GREEN - the fixture did not "
            "fire, so nothing would ever notice a worker that missed the redirect")
        out = result.stdout.str()
        assert "intended" in out, "the failure must say what was expected"
        assert "localm_test_home_" in out, "must name the directory it kept"
        left = glob.glob(str(sub_temp_root / "localm_test_home_*"))
        assert left, "the evidence directory was removed anyway"

    def test_a_correctly_placed_run_stays_green_and_leaves_nothing(
            self, pytester, sub_temp_root, monkeypatch):
        """The fires-control for the test above. Same wiring, same env var, only
        pointed at the root the sub-run really uses."""
        monkeypatch.setenv("LOCALM_TEST_EXPECTED_TEMP_ROOT", str(sub_temp_root))
        _with_real_conftest(pytester)
        pytester.makepyfile("def test_ordinary():\n    assert True\n")
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=1)
        assert result.ret == 0
        assert glob.glob(str(sub_temp_root / "localm_test_home_*")) == [], (
            "a healthy run littered the temp root - this is the leak")

    def test_only_the_first_test_is_failed(self, pytester, sub_temp_root,
                                           monkeypatch):
        """A misplaced worker still ran its tests correctly; it just wrote to the
        wrong drive. Failing all of them would bury the run's real results, so
        the complaint is made once."""
        monkeypatch.setenv("LOCALM_TEST_EXPECTED_TEMP_ROOT",
                           str(sub_temp_root / "not-where-we-landed"))
        _with_real_conftest(pytester)
        pytester.makepyfile(
            "def test_a():\n    assert True\n"
            "def test_b():\n    assert True\n"
            "def test_c():\n    assert True\n")
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        assert result.ret != 0
        result.assert_outcomes(passed=2, errors=1)
        for path in glob.glob(str(sub_temp_root / "localm_test_home_*")):
            os.rmdir(path)


@pytest.mark.skipif(importlib.util.find_spec("xdist") is None,
                    reason="pytest-xdist is not installed, so there is no "
                           "controller-to-worker channel to exercise")
class TestTheExpectationReachesTheWorkersUnderXdist:
    """The claim the whole design rests on: the expectation travels over
    execnet's workerinput channel rather than the environment, so a worker that
    lost the env still gets judged. Proven with a real ``-n 2`` sub-run, because
    nothing short of one exercises controller-to-worker delivery."""

    def test_workers_receive_the_controllers_root_and_all_of_them_clean_up(
            self, pytester, sub_temp_root):
        _with_real_conftest(pytester, node_hook=True)
        pytester.makepyfile(
            "import os\n"
            f"ROOT = r'{sub_temp_root}'\n"
            "def test_the_worker_was_told_where_it_should_be(request):\n"
            "    workerinput = getattr(request.config, 'workerinput', None)\n"
            "    assert workerinput is not None, 'not running under xdist'\n"
            "    stamped = workerinput.get('localm_expected_temp_root')\n"
            "    assert stamped, 'the controller stamped nothing'\n"
            "    assert os.path.normcase(stamped) == os.path.normcase(ROOT)\n")
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider",
                                               "-n", "2")
        result.assert_outcomes(passed=1)
        assert result.ret == 0
        # The controller and both workers each made one; every one must be gone.
        assert glob.glob(str(sub_temp_root / "localm_test_home_*")) == [], (
            "an xdist run littered the temp root - one directory per process is "
            "exactly how 426 of them accumulated")

    def test_the_controller_stamps_its_own_resolved_root(self, tmp_path):
        """Unit-level companion to the run above, so a delivery failure and a
        wrong-value failure are distinguishable rather than one red blob."""
        root = tmp_path / "temproot"
        home, out = _run_probe(tmp_path, "stamp", temp_root=root)
        assert f"STAMPED={root}" in out.replace("/", os.sep), out
        assert not os.path.exists(home)


def _this_sessions_conftest(request):
    """The conftest module object THIS session is actually running.

    Taken from the plugin manager rather than ``import conftest``, which would
    guess at a module name, and rather than a fresh exec by path, which would
    load a second copy and assert about that instead - the copy could be healthy
    while the live one sat inert."""
    for plugin in request.config.pluginmanager.get_plugins():
        filename = getattr(plugin, "__file__", None) or ""
        if (filename.replace("\\", "/") == _REAL_CONFTEST
                and hasattr(plugin, "_test_home_dir")):
            return plugin
    raise AssertionError(
        "tests/conftest.py is not a registered plugin in this session, so none "
        "of its guards are live and every test in this file is vacuous")


class TestThisSessionsOwnGuardIsArmed:
    """The anti-theater assertion. Without it every test above could pass while
    the shipped conftest sat inert in the run that actually matters."""

    def test_the_running_session_resolved_a_temp_root(self, request):
        conftest = _this_sessions_conftest(request)
        assert conftest._actual_temp_root
        assert os.path.dirname(conftest._test_home_dir) == conftest._actual_temp_root
        assert os.path.isdir(conftest._test_home_dir), (
            "this session's own LOCALM_HOME is missing while the session is "
            "still running")

    def test_the_running_session_is_not_sitting_on_an_unreported_mismatch(
            self, request):
        """If this session's temp root were wrong, the autouse fixture would have
        failed the first test in this process rather than this one, so asserting
        it here makes the expectation legible instead of implicit."""
        conftest = _this_sessions_conftest(request)
        assert conftest._wrong_temp_root is None, conftest._wrong_temp_root

    def test_the_cleanup_handler_is_registered_in_this_session(self, request):
        """atexit exposes no API for listing its callbacks, so the registration
        itself is pinned statically, against the SHIPPED file. That is weak on
        its own, which is why it is not the proof: the subprocess tests above are
        (they watch a real process exit and find the directory gone, and watch
        the paired run without the handler and find it still there). This one
        exists so that deleting the register() line fails HERE, naming the cause,
        instead of only showing up as a leak nobody counts for another month."""
        conftest = _this_sessions_conftest(request)
        assert callable(conftest._cleanup_test_home)
        source = Path(_REAL_CONFTEST).read_text(encoding="utf-8")
        assert "atexit.register(_cleanup_test_home)" in source, (
            "the shipped conftest no longer arms the atexit cleanup, so every "
            "process that merely imports it leaks its temp home again")
