# SPDX-License-Identifier: AGPL-3.0-or-later
"""The system-path-touch guard must actually FIRE, and must not false-positive.

conftest's system-path guard exists because a test that reads, stats, or lists a
REAL system location has stopped testing our code: it depends on the machine it
runs on.

This drives the SHIPPED guard end to end through real sub-pytest runs, in two
halves:

  (i)  the recording machinery, through a benign marker root inside the test's own
       tmp_path - so the fires-control never goes near a real system path itself;
  (ii) the matcher, through pure string assertions with no filesystem at all.

Plus the assertion that the guard is ARMED in this very session, without which
every test here could pass while the shipped guard sat inert.

The sub-runs are SUBPROCESS runs: this session's own guard is armed in-process,
and an in-process sub-run would record the sub-test's deliberate touch into the
outer session's hit table.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REAL_CONFTEST = str(Path(__file__).parent / "conftest.py").replace("\\", "/")


def _real_conftest_module():
    """Load the SHIPPED tests/conftest.py under its own module name.

    By path, not by ``import``, so these assert against the real file, and under
    a distinct module name because the generated sub-conftest is itself called
    ``conftest``."""
    spec = importlib.util.spec_from_file_location(
        "_localm_syspath_conftest", _REAL_CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_real_conftest(pytester):
    """Point the sub-run at the REAL tests/conftest.py.

    Only the two names the guard needs are re-exported. A ``from conftest import
    *`` skips every underscore-prefixed name, so the guard would never arrive and
    the sub-run would pass with nothing installed.
    """
    pytester.makeconftest(
        "import importlib.util as _u, sys\n"
        "_s = _u.spec_from_file_location('_localm_real_conftest', r'"
        + _REAL_CONFTEST + "')\n"
        "_m = _u.module_from_spec(_s)\n"
        "sys.modules['_localm_real_conftest'] = _m\n"
        "_s.loader.exec_module(_m)\n"
        "_no_system_path_touches = _m._no_system_path_touches\n"
        "pytest_sessionfinish = _m.pytest_sessionfinish\n"
    )


@pytest.fixture
def marker_root(tmp_path, monkeypatch):
    """A benign, disposable directory that the guard is told to treat as a system
    location, via the ADD-only extra-markers env var.

    The machinery under test is identical; only the marker list differs. The env
    var can only ADD roots, so a sub-run can never use it to weaken the shipped
    guard."""
    root = tmp_path / "pretend-system-root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "thing.cfg").write_text("data\n", encoding="utf-8")
    monkeypatch.setenv("LOCALM_TEST_SYSPATH_EXTRA_MARKERS", str(root))
    return root


class TestTheGuardIsActuallyArmed:
    """The guard must be armed in this session."""

    def test_the_filesystem_entry_points_are_wrapped_in_this_session(self):
        """Not 'a guard exists somewhere' but 'this run's os.stat is the guard's'.

        Checked through __qualname__ because the wrapper is a closure defined
        inside _arm_system_path_guard, so this cannot pass against the unpatched
        stdlib function."""
        import io
        wrapped = {
            "os.stat": os.stat, "os.lstat": os.lstat, "os.scandir": os.scandir,
            "os.listdir": os.listdir, "os.path.realpath": os.path.realpath,
            "io.open": io.open,
        }
        for label, func in wrapped.items():
            assert "guarded" in getattr(func, "__qualname__", ""), (
                f"{label} is not wrapped: the system-path guard is NOT armed in "
                "this session, so every syspath test is vacuous")

    def test_a_real_system_root_is_among_the_markers(self):
        """The guard must be watching something real, derived at runtime and not
        hardcoded. On Windows that is %SystemRoot%; elsewhere the FHS config
        roots. An empty marker list makes the guard inert."""
        conftest = _real_conftest_module()
        roots = conftest._syspath_marker_roots()
        assert roots, "no system-path markers at all"
        if os.name == "nt":
            expected = (os.environ.get("SystemRoot")
                        or os.environ.get("windir") or "")
            assert expected, "no SystemRoot/windir in the environment to derive from"
            assert expected.replace("\\", "/").rstrip("/").lower() in roots
        else:
            assert "/etc" in roots


class TestTheMatcher:
    """Half (ii): pure strings, no filesystem touched at all - not even a benign
    one."""

    @pytest.fixture
    def rx(self):
        return _real_conftest_module()._syspath_regex(
            ["z:/pretend", "/etc", "/boot"])

    @pytest.mark.parametrize("path,expected", [
        ("z:/pretend", True),
        ("z:/pretend/sub/x.cfg", True),
        ("Z:\\Pretend\\sub", True),          # backslashes normalise
        ("Z:/PRETEND/x", True),              # and case folds
        ("/etc", True),
        ("/etc/sub/x.conf", True),
        ("z:/pretendtoo/x", False),          # segment boundary, not raw prefix
        ("/etcetera/x", False),
        ("etc/x", False),                    # relative lookalike
        ("d:/projects/localm/x.py", False),
        ("d:/x/etc/y", False),               # marker must be at the START
    ])
    def test_marker_matching(self, rx, path, expected):
        assert _real_conftest_module()._syspath_matches(rx, path) is expected

    def test_non_path_inputs_do_not_explode(self, rx):
        """os.stat legitimately accepts an int fd, which is not a path at all, and
        the matcher must not raise on one."""
        conftest = _real_conftest_module()
        assert conftest._syspath_matches(rx, 3) is False
        assert conftest._syspath_matches(rx, b"/etc/x") is True
        assert conftest._syspath_matches(None, "/etc/x") is False

    def test_extra_markers_can_only_add(self, monkeypatch):
        """The env var can only ADD roots; it cannot switch the guard off."""
        conftest = _real_conftest_module()
        baseline = conftest._syspath_marker_roots()
        # The var is os.pathsep-separated, so the value itself must not contain the
        # separator; the fixture value has to be platform-appropriate.
        extra = "d:/somewhere/else" if os.name == "nt" else "/somewhere/else"
        assert os.pathsep not in extra, "the marker value must survive the split"
        monkeypatch.setenv("LOCALM_TEST_SYSPATH_EXTRA_MARKERS", extra)
        widened = conftest._syspath_marker_roots()
        assert set(baseline) <= set(widened), "markers were removed, not added"
        assert extra in widened

    def test_extra_markers_split_on_the_platform_separator(self, monkeypatch):
        """Two markers in one value must arrive as two roots, not one, so the
        separator itself is what is under test.
        """
        conftest = _real_conftest_module()
        one = "/alpha/one" if os.name != "nt" else "y:/alpha/one"
        two = "/beta/two" if os.name != "nt" else "z:/beta/two"
        monkeypatch.setenv("LOCALM_TEST_SYSPATH_EXTRA_MARKERS",
                           one + os.pathsep + two)
        widened = conftest._syspath_marker_roots()
        assert one in widened and two in widened


class TestTheRecordingMachineryFires:
    """Half (i): every patched entry point, driven through a real sub-pytest run
    against a benign marker root. Parametrised over the entry points because they
    are patched individually. io.open and builtins.open are the same function
    object but two separate module attributes: bare open() goes through one,
    Path.read_text() through the other, so patching either alone leaves a
    hole."""

    @pytest.mark.parametrize("body,label", [
        ("os.stat(TARGET)", "os.stat"),
        ("os.lstat(TARGET)", "os.lstat"),
        ("os.listdir(SUBDIR)", "os.listdir"),
        ("list(os.walk(ROOT))", "os.scandir via os.walk"),
        ("os.path.realpath(TARGET)", "os.path.realpath"),
        ("Path(TARGET).exists()", "Path.exists -> os.stat"),
        ("Path(TARGET).resolve()", "Path.resolve -> realpath"),
        ("Path(TARGET).read_text(encoding='utf-8')", "Path.read_text -> io.open"),
        ("open(TARGET, encoding='utf-8').close()", "bare open -> builtins.open"),
        ("list(Path(SUBDIR).glob('*'))", "Path.glob -> os.scandir"),
    ])
    def test_a_touch_is_caught_and_fails_the_run(self, pytester, marker_root,
                                                 body, label):
        _with_real_conftest(pytester)
        pytester.makepyfile(
            "import os, os.path\n"
            "from pathlib import Path\n"
            f"ROOT = r'{marker_root}'\n"
            f"SUBDIR = r'{marker_root / 'sub'}'\n"
            f"TARGET = r'{marker_root / 'sub' / 'thing.cfg'}'\n"
            "def test_touches_a_marked_path():\n"
            f"    {body}\n"
        )
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        assert result.ret != 0, (
            f"a test touching a marked path via {label} exited GREEN - the guard "
            "did not catch this entry point")
        result.stdout.fnmatch_lines(["*touched a real system path*"])

    def test_the_report_names_the_path_the_frame_and_the_fix(self, pytester,
                                                             marker_root):
        _with_real_conftest(pytester)
        pytester.makepyfile(
            "import os\n"
            f"TARGET = r'{marker_root / 'sub' / 'thing.cfg'}'\n"
            "def test_touches_a_marked_path():\n"
            "    os.stat(TARGET)\n"
        )
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        out = result.stdout.str()
        assert "thing.cfg" in out, "must name the path that was touched"
        assert "test_the_report_names" in out or "test_touches_a_marked_path" in out, \
            "must attribute the touch to the test that made it"
        assert "tmp_path" in out, "must point at the fix, not just complain"


class TestTheGuardExemptsTheRealCertStoreProbe:
    """ssl.create_default_context() (verified_urlopen's native-store path,
    localm/http_ssl.py) reads a small, fixed set of read-only trust-anchor
    paths under /etc on POSIX - real filesystem access, but categorically
    different from the credentials/config /etc otherwise marks: a CA bundle
    is data every TLS client on the machine needs to read.

    Measured directly against CPython 3.12 on Ubuntu 24.04 (matching CI):
    patching os.stat and calling create_default_context() records exactly
    /etc/ssl/cert.pem, /etc/pki/tls/cert.pem, /etc/ssl/certs, and nothing
    else - the allowlist in conftest.py was built from that measurement,
    not assumed from distro folklore."""

    @pytest.mark.skipif(os.name == "nt",
                        reason="the /etc marker, and this probe, are POSIX-only")
    def test_ssl_create_default_context_does_not_trip_the_guard(self, pytester):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import ssl

            def test_native_cert_store_lookup():
                ssl.create_default_context()
        """)
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        assert result.ret == 0, (
            "a legitimate ssl.create_default_context() call tripped the "
            "system-path guard:\n" + result.stdout.str())
        result.assert_outcomes(passed=1)

    @pytest.mark.skipif(os.name == "nt",
                        reason="the /etc marker is POSIX-only")
    def test_the_exemption_stays_narrow_an_unrelated_etc_path_still_fires(
            self, pytester):
        """The fix must not widen into "anything under /etc/ssl is fine" -
        an unrelated /etc path the allowlist does not name is still caught."""
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import os

            def test_touches_an_unrelated_etc_path():
                try:
                    os.stat("/etc/hostname")
                except FileNotFoundError:
                    pass
        """)
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        assert result.ret != 0, (
            "the cert-store exemption leaked into an unrelated /etc path")
        result.stdout.fnmatch_lines(["*touched a real system path*"])


class TestTheGuardDoesNotFalsePositive:
    """The quiet cases: paths that must NOT be flagged."""

    def test_ordinary_tmp_path_work_is_fine(self, pytester, marker_root):
        _with_real_conftest(pytester)
        pytester.makepyfile("""
            import os
            from pathlib import Path

            def test_stays_in_its_own_tmp_path(tmp_path):
                d = tmp_path / "work" / "src"
                d.mkdir(parents=True)
                f = d / "a.py"
                f.write_text("# hello\\n", encoding="utf-8")
                assert f.exists()
                assert f.read_text(encoding="utf-8").startswith("#")
                assert f.resolve().is_file()
                assert list(d.glob("*.py")) == [f]
                assert os.listdir(d) == ["a.py"]
                list(os.walk(tmp_path))
        """)
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=1)
        assert result.ret == 0

    def test_a_sibling_directory_sharing_a_marker_prefix_is_fine(
            self, pytester, marker_root):
        """``pretend-system-root-2`` is not inside ``pretend-system-root``. A
        raw prefix test would flag it; the segment-boundary match must not."""
        sibling = Path(str(marker_root) + "-2")
        (sibling).mkdir()
        (sibling / "f.txt").write_text("x\n", encoding="utf-8")
        _with_real_conftest(pytester)
        pytester.makepyfile(
            "import os\n"
            f"TARGET = r'{sibling / 'f.txt'}'\n"
            "def test_sibling_is_not_the_marked_root():\n"
            "    os.stat(TARGET)\n"
        )
        result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=1)
        assert result.ret == 0
