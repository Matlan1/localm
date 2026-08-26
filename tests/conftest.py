# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pytest fixtures.

Hermetic data dir: every test gets its own ``LOCALM_HOME`` under the test's
``tmp_path``. Without this, ``_detect_home()`` falls through to *portable mode*
(a ``home/`` directory next to the installed package) - so a developer who has
run the GUI in portable mode would have the GUI tests read, write, and DELETE
their actual conversations/images while the suite runs.
``LOCALM_HOME`` takes priority over portable mode in ``_detect_home()``, so
pinning it here isolates every test. Tests that need a specific home override
this with their own ``monkeypatch.setenv`` (which runs after this autouse
fixture).

That import-time directory is removed at PROCESS EXIT, not only at the end of a
pytest session, because this file is also loaded outside one - see the temp-root
section below for why the removal is conditional rather than unconditional.
"""

import atexit
import builtins
import io
import mimetypes
import os
import re
import stat as _stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import shutil
from pathlib import Path

# Isolate LOCALM_HOME globally at import time so that any module importing
# localm.config during test collection or execution resolves HOME_DIR to a
# temporary directory instead of the developer's real home config/keys.
_test_home_dir = tempfile.mkdtemp(prefix="localm_test_home_")
os.environ["LOCALM_HOME"] = _test_home_dir

# Same protection for the media-plugin legacy-workflow migration: on startup it
# MOVES a personal override OUT of the in-package source dir (localm/image_gen/
# flux_workflow.json etc.) into home/workflows. That source is the real repo
# checkout, NOT under the tmp LOCALM_HOME above, so letting it run during the
# suite - especially inside a localm SUBPROCESS a test spawns, where "pytest" is
# not in sys.modules - would move a developer's real workflow out of their
# working tree. This flag (inherited by spawned subprocesses) disables it
# everywhere; the migration logic is exercised directly in test_media_workflows.
os.environ["LOCALM_SKIP_LEGACY_WORKFLOW_MIGRATION"] = "1"


def pytest_sessionfinish(session, exitstatus):
    _cleanup_test_home()
    _report_system_path_touches(session)
    _report_installer_runs(session)
    _report_wrong_temp_root(session)


import pytest


# --------------------------------------------------------------------------- #
#  The import-time temp home is cleaned up at PROCESS EXIT - except when it     #
#  landed on the wrong temp root, which is the one case worth keeping           #
#                                                                              #
#  The mkdtemp above runs at IMPORT time (deliberately: see the module          #
#  docstring), so it runs once per PROCESS - including every process that loads #
#  this file WITHOUT finishing a pytest session. tests/test_conftest_disk_guard #
#  and tests/test_conftest_syspath_guard both exec it by path under their own   #
#  module name, and a plain script that unit-checks one of those guards does    #
#  the same. Cleaning up only in pytest_sessionfinish therefore leaked one      #
#  directory per such process, permanently: 426 of them had piled up in one     #
#  developer's temp dir by 2026-07-22. atexit closes that.                     #
#                                                                              #
#  An UNCONDITIONAL atexit would have closed it by breaking something else. A   #
#  test run redirects the temp root to a scratch drive by setting               #
#  TMPDIR/TEMP/TMP, and an xdist WORKER can end up not using it (a tempfile     #
#  module that already cached its answer, an env block that did not arrive).    #
#  Because the mkdtemp is per process, a worker that missed the redirect        #
#  necessarily left its directory on the WRONG root, so counting leftovers      #
#  there answered "did the redirect reach every worker?" - and removing every   #
#  directory on exit would make that count read zero whether or not one missed  #
#  it. Silently always-green is worse than no check at all: it is the same      #
#  failure shape as a guard that never arms.                                   #
#                                                                              #
#  So the cleanup is CONDITIONAL on landing where the run intended:             #
#    landed right -> removed, so nothing leaks and a leftover count reads zero; #
#    landed wrong -> KEPT, and the run goes red naming the directory.           #
#  A leftover count keeps its exact old meaning (zero == every process used the #
#  intended root) while the healthy case stops littering, and the anomaly is    #
#  now loud at the time it happens instead of a forensic exercise afterwards.   #
#                                                                              #
#  The intent reaches a worker over execnet's workerinput channel, NOT the      #
#  environment. That is the whole point: the environment is the thing under     #
#  suspicion, so a worker that lost it would lose an env-carried expectation    #
#  too and go quiet, which is the always-green failure again. xdist calls       #
#  pytest_configure_node on the controller and sends the (mutated) workerinput  #
#  over its own channel, so the expectation arrives even when the env did not.  #
# --------------------------------------------------------------------------- #

_EXPECTED_TEMP_ROOT_KEY = "localm_expected_temp_root"
_EXPECTED_TEMP_ROOT_ENV = "LOCALM_TEST_EXPECTED_TEMP_ROOT"

# Where this process's temp root actually resolved. Read off the directory
# mkdtemp just made rather than by calling gettempdir() again: gettempdir()
# caches its first answer, and what matters here is where the bytes really went,
# not what a later call would say.
_actual_temp_root = os.path.dirname(_test_home_dir)

_expected_temp_root = None      # the root the run intended; None = none declared
_wrong_temp_root = None         # the complaint, once a mismatch is established
_wrong_temp_root_reported = False

_WRONG_TEMP_ROOT_ADVICE = (
    "  The directory named above was LEFT on disk on purpose: it is the evidence,\n"
    "  and counting leftovers is how a run proves every one of its processes used\n"
    "  the intended temp root. Export TMPDIR/TEMP/TMP BEFORE starting pytest -\n"
    "  setting them afterwards is too late, because tempfile caches the first\n"
    "  answer it resolves - then delete that one directory, by name.")


def _same_dir(a, b) -> bool:
    """True when two paths name the same directory.

    Normalised strings first, which is the entire answer whenever both sides
    were resolved from the same TEMP value. Only when THAT disagrees is realpath
    consulted, so a mere difference in spelling (an 8.3 short name, a junction)
    is not reported as a misplaced run. The realpath call touches the filesystem,
    which is why it is the second question and not the first: in a healthy run it
    never happens."""
    def norm(p):
        return os.path.normcase(os.path.abspath(p)).replace("\\", "/").rstrip("/")
    if norm(a) == norm(b):
        return True
    try:
        return norm(os.path.realpath(a)) == norm(os.path.realpath(b))
    except OSError:
        return False


def pytest_configure(config):
    """Work out which temp root this run intended, and judge where we landed."""
    global _expected_temp_root, _wrong_temp_root
    workerinput = getattr(config, "workerinput", None)
    # An env-declared root can only ADD a check to a run that had none: a value
    # stamped by the controller always wins, so this seam (which the guard's own
    # fires-controls use) can never relax or redirect the xdist check. Same
    # add-only contract as LOCALM_TEST_SYSPATH_EXTRA_MARKERS, for the same reason.
    from_env = os.environ.get(_EXPECTED_TEMP_ROOT_ENV) or None
    if workerinput is not None:
        _expected_temp_root = workerinput.get(_EXPECTED_TEMP_ROOT_KEY) or from_env
        if _expected_temp_root is None:
            # We are a worker and nobody told us what the run intended, so the
            # check below cannot arm. Say so instead of quietly cleaning up: an
            # oracle that silently does not run is the exact failure this whole
            # section exists to prevent, and this is the only way it can happen.
            _wrong_temp_root = (
                "this xdist worker was never told which temp root the run "
                "intended, so the temp-root check is NOT ARMED and a worker that "
                "missed the redirect would go unnoticed. The controller did not "
                "load tests/conftest.py, which is where pytest_configure_node "
                f"stamps {_EXPECTED_TEMP_ROOT_KEY!r}. Invoke pytest so that tests/ "
                "is among its arguments (testpaths does this already), or set "
                f"{_EXPECTED_TEMP_ROOT_ENV} to the intended temp root.")
            return
    else:
        _expected_temp_root = from_env
    if _expected_temp_root and not _same_dir(_actual_temp_root, _expected_temp_root):
        _wrong_temp_root = (
            f"this process resolved its temp root to {_actual_temp_root} but the "
            f"run intended {_expected_temp_root}, so everything it wrote through "
            "tempfile went to the wrong place, including "
            f"{_test_home_dir}")


try:
    import xdist                                        # noqa: F401
except ImportError:                                     # xdist is optional
    pass
else:
    def pytest_configure_node(node):
        """Hand each worker the temp root the run intended, over execnet's own
        channel rather than the environment (see the section comment).

        The controller's OWN resolved root is the fallback expectation: with
        nothing declared, "the root the controller used" is exactly what a worker
        has to match, which is the question the leftover count used to answer.
        When a root WAS declared, that wins, so a controller which itself missed
        the redirect does not quietly become the standard its workers are held
        to - it gets reported on its own account instead."""
        node.workerinput[_EXPECTED_TEMP_ROOT_KEY] = (
            _expected_temp_root or _actual_temp_root)


def _cleanup_test_home():
    """Remove the import-time LOCALM_HOME, unless it is evidence.

    Idempotent: pytest_sessionfinish calls it at the end of a session and atexit
    catches every other way a process can end (including having no session at
    all, which is the leak this exists to close)."""
    if _wrong_temp_root:
        return                  # KEEP it - this directory IS the evidence
    try:
        shutil.rmtree(_test_home_dir)
    except FileNotFoundError:
        pass                    # already gone: sessionfinish ran, then atexit
    except OSError as exc:
        # Not silenced. A directory we could not remove is precisely the leak
        # this function exists to stop, and swallowing the failure would rebuild
        # that pile one run at a time with nothing to show for it.
        #
        # Written to the ORIGINAL stderr, and guarded, because this also runs
        # from atexit: by then pytest's capture replacement can be torn down, and
        # a print that raises there is reported as "Error in
        # atexit._run_exitfuncs" with a traceback that buries the actual message.
        # If even that stream is gone there is nowhere left to report to, which
        # is the one case where nothing is the honest outcome.
        try:
            print(f"WARNING: could not remove the test LOCALM_HOME "
                  f"{_test_home_dir}: {exc}",
                  file=sys.__stderr__ or sys.stderr, flush=True)
        except Exception:
            pass


atexit.register(_cleanup_test_home)


@pytest.fixture(autouse=True)
def _temp_root_is_the_one_the_run_intended():
    """Fail once, in the process that got its temp root wrong.

    Once per PROCESS rather than once per test on purpose: under -n auto every
    test in a misplaced worker would otherwise fail, and those tests are not
    wrong - they exercised the code correctly, they just wrote their bytes to the
    wrong drive. One named failure is unmissable without burying the run's real
    results under a few thousand identical ones.

    A fixture rather than only pytest_sessionfinish because a worker's exitstatus
    does not become the run's under -n auto - the same reason the system-path
    guard is enforced per test - but a failed test does."""
    global _wrong_temp_root_reported
    if _wrong_temp_root and not _wrong_temp_root_reported:
        _wrong_temp_root_reported = True
        pytest.fail(_wrong_temp_root + "\n" + _WRONG_TEMP_ROOT_ADVICE)


def _report_wrong_temp_root(session):
    """Session-level backstop for the process the fixture cannot reach.

    Under -n auto the CONTROLLER runs no tests at all, so a controller that
    itself missed the redirect would leave its directory behind with nothing to
    say why. The controller's exitstatus IS the run's, so failing here covers
    exactly the case the per-test fixture cannot - the mirror image of
    _report_system_path_touches, which is a no-op on the controller for the
    opposite reason."""
    if not _wrong_temp_root:
        return
    print("\nWRONG TEMP ROOT:\n  " + _wrong_temp_root + "\n"
          + _WRONG_TEMP_ROOT_ADVICE)
    if getattr(session, "exitstatus", 0) == 0:
        session.exitstatus = 1


# --------------------------------------------------------------------------- #
#  No test may touch a real system path                                         #
#                                                                              #
#  A test's business is the code under test and its own tmp_path. When a test   #
#  reads, stats, or lists a REAL system location it stops being a test of our   #
#  code: it depends on the machine it runs on, it differs per contributor and   #
#  per CI image, and - the reason this is enforced rather than asked for - it   #
#  is the same shape as the production defect it is usually written to cover.   #
#  The coder's scope gate used to resolve() a model-supplied absolute path in   #
#  order to REFUSE it, so refusing reached out and stat-ed whatever the model   #
#  named, anywhere on the machine (#802 for the shell warning path, and the     #
#  enforcement path after it). Tests that name a real system file teach that    #
#  the touch is normal, and four confinement test files had to be purged of     #
#  exactly that. A guard makes the next one fail instead of merging.            #
#                                                                              #
#  Syscall patching, not sys.addaudithook: MEASURED on 3.12.13, exists(),       #
#  resolve() and os.stat() emit NO audit event at all (only open and            #
#  os.listdir do), so an audit hook is blind to precisely the calls that        #
#  matter. Both `builtins.open` AND `io.open` are patched: they are the same    #
#  function object but two separate module attributes, and (measured) bare      #
#  open() goes through the first while Path.read_text()/Path.open() go through  #
#  the second, so patching either one alone leaves a hole.                      #
#                                                                              #
#  Marker roots are DERIVED, never a hardcoded drive literal: the Windows       #
#  system root comes from %SystemRoot%/%windir% at runtime. The guard only ever #
#  compares STRINGS - it never opens, stats, or reads any system path itself.   #
#                                                                              #
#  Deliberately NOT marked: /proc, /sys, /dev and /usr. Those are runtime       #
#  interfaces with legitimate library traffic (psutil reads /proc constantly),  #
#  so marking them would make the guard cry wolf until someone silenced it.     #
#  The markers are config/credential roots, which is where the defect lives.    #
# --------------------------------------------------------------------------- #

_SYSPATH_EXTRA_ENV = "LOCALM_TEST_SYSPATH_EXTRA_MARKERS"

# (kind, path, origin frame) -> count. Module-level so a per-test fixture can
# diff it and the session report can summarise it.
_SYSPATH_HITS: dict = {}
_SYSPATH_ARMED = False
_GUARD_FILE = __file__.replace("\\", "/")


def _syspath_marker_roots() -> list:
    """The path prefixes that count as a real system location, lowercased and
    slash-normalised.

    ``_SYSPATH_EXTRA_ENV`` ADDS roots and can never remove one, so the fires-
    control (which points it at a benign directory inside the test's own
    tmp_path) can prove the machinery works without weakening the guard and
    without any test going near a real system path."""
    roots = []
    for var in ("SystemRoot", "windir"):
        value = os.environ.get(var)
        if value:
            roots.append(value)
    if os.name != "nt":
        # FHS config/credential roots. See the section comment for why the
        # runtime interfaces (/proc, /sys, /dev) are deliberately absent.
        roots += ["/etc", "/boot", "/root"]
    roots += [p for p in os.environ.get(_SYSPATH_EXTRA_ENV, "").split(os.pathsep)
              if p.strip()]
    out, seen = [], set()
    for raw in roots:
        norm = raw.replace("\\", "/").rstrip("/").lower()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _syspath_regex(roots):
    """Anchored alternation over *roots*, or None when there is nothing to match.

    The trailing ``(?:/|$)`` makes the match land on a path SEGMENT boundary, so
    a marker of ``/etc`` flags ``/etc`` and ``/etc/x`` but not ``/etcetera``."""
    if not roots:
        return None
    return re.compile("(?:" + "|".join(re.escape(r) for r in roots) + r")(?:/|$)")


def _syspath_matches(rx, raw) -> bool:
    """True when *raw* names a marked system location. Pure string work: this
    never touches the filesystem, and tolerates anything os.stat accepts
    (str, bytes, PathLike, or an int fd, which is not a path at all)."""
    if rx is None:
        return False
    try:
        s = os.fspath(raw)
    except TypeError:
        return False                      # an int fd or similar: not a path
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    elif not isinstance(s, str):
        return False
    return rx.match(s.replace("\\", "/").lower()) is not None


def _syspath_origin() -> str:
    """The nearest tests/ or localm/ frame, so the report names the code that
    reached out rather than the stdlib helper that happened to do the syscall."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename.replace("\\", "/")
        if filename != _GUARD_FILE and ("/tests/" in filename
                                        or "/localm/" in filename):
            return f"{filename}:{frame.f_lineno}"
        frame = frame.f_back
    return "<no tests/ or localm/ frame>"


def _arm_system_path_guard() -> bool:
    """Wrap the filesystem entry points. Returns True when the guard is live.

    The hot path is a single anchored regex match against a normalised string;
    the expensive part (walking the stack for the originating frame) runs ONLY
    after a marker has already matched, which on a healthy suite is never."""
    global _SYSPATH_ARMED
    if _SYSPATH_ARMED:
        return True
    rx = _syspath_regex(_syspath_marker_roots())
    if rx is None:
        return False

    def _wrap(kind, func):
        def guarded(path, *args, **kwargs):
            if _syspath_matches(rx, path):
                key = (kind, str(path), _syspath_origin())
                _SYSPATH_HITS[key] = _SYSPATH_HITS.get(key, 0) + 1
            return func(path, *args, **kwargs)
        return guarded

    os.stat = _wrap("os.stat", os.stat)
    os.lstat = _wrap("os.lstat", os.lstat)
    os.scandir = _wrap("os.scandir", os.scandir)
    os.listdir = _wrap("os.listdir", os.listdir)
    os.path.realpath = _wrap("os.path.realpath", os.path.realpath)
    builtins.open = _wrap("open", builtins.open)
    io.open = _wrap("io.open", io.open)
    _SYSPATH_ARMED = True
    return True


# Warm the stdlib mimetypes registry BEFORE arming the guard, deliberately.
#
# This is NOT a localm defect being papered over, and the distinction is the
# whole reason it is done here rather than silenced at the reporting end.
# Anything that maps a file EXTENSION to a content type must consult the OS mime
# registry, and the stdlib does that lazily on first use: on POSIX it reads
# /etc/mime.types plus the httpd/apache paths. Measured directly -
# mimetypes.guess_type("app.js"), with no add_type call anywhere, touches /etc
# six times. Windows reads its registry instead, so the marker roots never match
# and this was invisible there, which is why it only ever failed on Linux.
#
# The read is therefore real, legitimate and unavoidable: it cannot be dropped
# without breaking content-type detection localm genuinely wants (localm/cli/
# chat.py and localm/plugins/gui/routes/share.py both guess_type user files
# against system mime data). What it must NOT do is land inside whichever test
# happens to construct an app first and be reported as that test reaching out -
# a true statement about the wrong subject. Doing it once, here, in a declared
# place, keeps a hit meaning what the guard says it means: OUR code touching a
# system location, not the interpreter initialising itself.
#
# guess_type(), NOT init(): this module is re-executed under an ALREADY-ARMED
# guard by test_conftest_syspath_guard's _real_conftest_module(), which loads the
# shipped conftest by path to assert against the real file. init() rebuilds the
# registry unconditionally, so it re-read /etc on every such re-exec and the
# guard duly reported it against that test - 15 teardown errors, self-inflicted.
# guess_type() goes through the stdlib's own "if _db is None: init()", so it is
# idempotent by construction and warms exactly once per process, which is also
# precisely the call production makes.
mimetypes.guess_type("warm.js")

_arm_system_path_guard()


def _format_syspath_hits(hits) -> str:
    return "\n".join(
        f"    {kind}({path})\n        from {origin}   x{count}"
        for (kind, path, origin), count in sorted(hits.items()))


_SYSPATH_ADVICE = (
    "  A test must not read, stat, or list a real system location. Create a real\n"
    "  but DISPOSABLE file or directory inside the test's own tmp_path and point\n"
    "  the code under test at that instead - it exercises the same path handling\n"
    "  without depending on the machine, and without teaching that reaching out is\n"
    "  normal. If a code path genuinely reached out on its own, that is the bug the\n"
    "  guard just found: fix the code, do not exempt the test.")


@pytest.fixture(autouse=True)
def _no_system_path_touches(request):
    """Fail the test that touched a real system path.

    Enforced per-test rather than only at session end because that is what works
    under ``-n auto``: a worker's session exitstatus does not become the run's,
    but a failed test does. It also names the culprit directly instead of leaving
    a pile of hits to attribute by hand. The session report below still runs, for
    anything recorded outside a test (import or collection time)."""
    before = dict(_SYSPATH_HITS)
    yield
    new = {k: v - before.get(k, 0) for k, v in _SYSPATH_HITS.items()
           if v > before.get(k, 0)}
    if new:
        pytest.fail("this test touched a real system path:\n"
                    + _format_syspath_hits(new) + "\n" + _SYSPATH_ADVICE)


def _report_system_path_touches(session):
    """Session-level backstop: report touches and fail the run.

    Covers what the per-test fixture cannot see - a touch at import or collection
    time, before any test started. Under xdist the controller records nothing of
    its own (the workers do the work and fail their own tests), so this is a
    no-op there rather than a second, conflicting verdict."""
    if not _SYSPATH_HITS:
        return
    print("\nSYSTEM PATH TOUCHES DETECTED (tests must stay inside tmp_path):\n"
          + _format_syspath_hits(_SYSPATH_HITS) + "\n" + _SYSPATH_ADVICE)
    if getattr(session, "exitstatus", 0) == 0:
        session.exitstatus = 1


# --------------------------------------------------------------------------- #
#  No test may install packages INTO THE INTERPRETER RUNNING THE SUITE         #
#                                                                              #
#  Found the hard way: windows CI went red at roughly 1 in 3 with              #
#  `AttributeError: module 'numpy' has no attribute 'asarray'` from            #
#  rag/store.py, on unrelated branches, with master green in the same window.  #
#  numpy is not a declared dependency and never appears in the install step.   #
#                                                                              #
#  Cause: a test reaches a REAL installer, captured live from the CI process   #
#  table as `uv pip install --python <the venv running the suite>              #
#  faster-whisper>=1.2.1`. numpy comes in transitively, and WHILE THE WHEEL IS #
#  UNPACKING `site-packages/numpy` exists with no `__init__.py` yet - which is #
#  a PEP 420 namespace package, so `import numpy` SUCCEEDS and returns a       #
#  module with no attributes. Any worker calling _cosine inside that window    #
#  gets the AttributeError. The install then finishes, __init__.py lands, and  #
#  the evidence erases itself, which is why it never reproduces afterwards.    #
#                                                                              #
#  Nothing keyed on the exception type can catch that: nothing raises. And the #
#  reason it is invisible on a developer box is worse - numpy is usually       #
#  already installed there, so the requirement is satisfied, no install runs,  #
#  and the window never opens. The bug hides wherever anyone would look for it.#
#                                                                              #
#  `deps._run_pip` passes `--python sys.executable`, so the install lands in   #
#  WHATEVER INTERPRETER THE SUITE IS ON: a throwaway venv in CI, but the       #
#  shared .venv when the suite is run locally - and #839 made the local suite  #
#  the load-bearing gate. So this is a rule breach with no author.             #
#                                                                              #
#  Installing into a DISPOSABLE venv under tmp_path stays allowed: the         #
#  managed-ComfyUI tests legitimately build one and pip into it. The line is   #
#  the TARGET, not the act, which is why this needs no allowlist to start with.#
# --------------------------------------------------------------------------- #

_INSTALLER_HITS: dict = {}
_INSTALLER_ARMED = False


def _norm_path(p) -> str:
    try:
        return os.path.normcase(os.path.abspath(str(p)))
    except (OSError, ValueError):
        return os.path.normcase(str(p))


def _installer_argv(cmd) -> list:
    if isinstance(cmd, (list, tuple)):
        return [str(a) for a in cmd]
    if isinstance(cmd, str):
        return cmd.split()
    return []


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0].lower()


# pip, pip3, pip3.12 - but NOT pipx, which is a different tool entirely.
_PIP_SHIM = re.compile(r"^pip[0-9.]*$")


def _installs_into_this_interpreter(cmd):
    """The reason string when *cmd* installs into the running interpreter, else None.

    Deliberately narrow. It must not fire on `pip cache dir`, `pip freeze`,
    `pip list` or `pip --version` (a real pip child that installs NOTHING is fine,
    and the cache-containment tests use exactly that), nor on an install aimed at
    ANOTHER interpreter - installing into a disposable venv under tmp_path is
    legitimate and the managed-ComfyUI tests do it.

    The bare-name case is the one this originally missed, and it is the form
    anyone would write by hand: `pip install x` has no directory component, so
    resolving it as a path lands on the CWD and matches neither sys.executable nor
    sys.prefix. But a bare `pip`/`uv` is found on PATH, and under the suite PATH
    leads to the venv running it - so no explicit target means THIS interpreter,
    not "some other one". (Gap and fix from local_7cb0d210, who hit the same class
    in their own version of this guard.)"""
    argv = _installer_argv(cmd)
    if not argv or "install" not in [a.lower() for a in argv]:
        return None

    head = _stem(argv[0])
    rest = [a.lower() for a in argv[1:]]
    target = None
    bare = os.path.dirname(argv[0]) == ""     # found via PATH, not a real path

    if head == "uv" and rest[:2] == ["pip", "install"]:
        target = argv[argv.index("--python") + 1] if "--python" in argv else None
    elif len(argv) >= 4 and argv[1] == "-m" and _stem(argv[2]) == "pip":
        if rest[2] != "install":
            return None                       # `-m pip cache dir`, `-m pip freeze`
        target, bare = argv[0], False         # explicit interpreter, judge by path
    elif _PIP_SHIM.match(head):
        if not rest or rest[0] != "install":
            return None                       # `pip list`, `pip --version`
        target = None if bare else argv[0]
    else:
        return None

    if target is None:
        # No explicit target: `uv pip install` without --python, or a bare pip
        # shim off PATH. Both resolve to the ACTIVE environment.
        target = os.environ.get("VIRTUAL_ENV") or sys.executable

    t = _norm_path(target)
    if t == _norm_path(sys.executable) or t.startswith(_norm_path(sys.prefix)):
        return "installs into the interpreter running the suite: " + " ".join(argv[:8])
    return None


def _arm_installer_guard() -> bool:
    """Patch subprocess.Popen once, at import.

    Popen rather than run(): run() goes through Popen, so one seam covers both,
    and `deps._run_pip` uses Popen directly. A test that fakes Popen for itself
    replaces this wrapper for that test and is therefore never flagged - correct,
    because a faked installer installs nothing."""
    global _INSTALLER_ARMED
    if _INSTALLER_ARMED:
        return False
    real_popen = subprocess.Popen

    class _GuardedPopen(real_popen):
        def __init__(self, args, *a, **kw):
            reason = _installs_into_this_interpreter(args)
            if reason:
                frames = []
                for frame in reversed(traceback.extract_stack()[:-1]):
                    f = frame.filename.replace("\\", "/")
                    if "/tests/" in f or "/localm/" in f:
                        frames.append(
                            f"{os.path.basename(frame.filename)}:{frame.lineno}")
                        if len(frames) >= 5:
                            break
                origin = frames[0] if frames else "unknown"
                # A spawn from a BACKGROUND THREAD may land while a different test
                # is current, so the nodeid would blame the wrong one - the very
                # misattribution this guard exists to remove, reproduced inside it.
                # It cannot be prevented (start_dep_install keeps the task, not the
                # thread), so it is made self-reporting: say so, and print the call
                # path, which IS attributable. Design from local_7cb0d210.
                thread = threading.current_thread()
                if thread is not threading.main_thread():
                    origin = (f"{origin}\n        NOT the main thread "
                              f"(thread {thread.name!r}): the test named above is "
                              f"whichever was CURRENT when this fired, not "
                              f"necessarily the one that STARTED it - attribute it "
                              f"from this call path instead:\n        "
                              + " <- ".join(frames))
                key = (reason, origin)
                _INSTALLER_HITS[key] = _INSTALLER_HITS.get(key, 0) + 1
                # BLOCK, do not merely record. Recording would still let the
                # install run, which is the whole harm: it mutates the suite's
                # own environment mid-run (and the developer's shared venv when
                # run locally), and the unpacking package is briefly importable
                # with no attributes, failing unrelated tests in other workers.
                # Reporting after the fact cannot undo either. The offending test
                # fails here AND is named by the fixture below.
                raise RuntimeError(
                    "BLOCKED by tests/conftest.py: " + reason + "\n"
                    + _INSTALLER_ADVICE)
            super().__init__(args, *a, **kw)

    subprocess.Popen = _GuardedPopen
    _INSTALLER_ARMED = True
    return True


_arm_installer_guard()


_INSTALLER_ADVICE = (
    "  A test must never install packages into the interpreter it is running on.\n"
    "  In CI that mutates the throwaway venv MID-RUN, and a package that is\n"
    "  unpacking is briefly importable with NO attributes (a directory with no\n"
    "  __init__.py is a namespace package), which fails unrelated tests in other\n"
    "  workers with errors that point nowhere near the cause. Run locally, the same\n"
    "  line mutates the developer's shared venv.\n"
    "  Fix the TEST, not this guard: stub the installer (monkeypatch subprocess.Popen\n"
    "  or deps._run_pip), drive the code path with dependency installation disabled,\n"
    "  or point the install at a DISPOSABLE venv under tmp_path - installing into one\n"
    "  of those is allowed and is not what this guard reports.")


def _format_installer_hits(hits) -> str:
    return "\n".join(f"    {reason}\n        from {origin}   x{count}"
                     for (reason, origin), count in sorted(hits.items()))


@pytest.fixture(autouse=True)
def _no_installer_into_this_interpreter(request):
    """Fail the test that spawned it, for the same reason the sibling guard does:
    under ``-n auto`` a worker's exitstatus does not become the run's, but a failed
    test does - and it names the culprit instead of leaving a global side effect to
    attribute by hand afterwards (which is exactly what made this take a day)."""
    before = dict(_INSTALLER_HITS)
    yield
    new = {k: v - before.get(k, 0) for k, v in _INSTALLER_HITS.items()
           if v > before.get(k, 0)}
    if new:
        pytest.fail("this test ran a package installer against the suite's own "
                    "interpreter:\n" + _format_installer_hits(new) + "\n"
                    + _INSTALLER_ADVICE)


def _report_installer_runs(session):
    """Backstop for an install at import or collection time, before any test ran.
    A no-op on the xdist controller, which spawns none of its own."""
    if not _INSTALLER_HITS:
        return
    print("\nPACKAGE INSTALLER RUN AGAINST THE SUITE'S OWN INTERPRETER:\n"
          + _format_installer_hits(_INSTALLER_HITS) + "\n" + _INSTALLER_ADVICE)
    if getattr(session, "exitstatus", 0) == 0:
        session.exitstatus = 1


# --------------------------------------------------------------------------- #
#  Resource-gated integration markers (V2)                                     #
#                                                                              #
#  A test tagged real_gguf / real_comfy / real_browser needs a real external  #
#  resource. Rather than each test re-implementing its own skip, gate them     #
#  centrally here: a tagged test is skipped (never failed) unless its resource #
#  is actually available, so the suite runs the real path the moment it is.    #
# --------------------------------------------------------------------------- #

def _runtime_available() -> bool:
    """True when the native llama.cpp runtime is provisioned and loadable."""
    try:
        from localm.inference.backends.llamacpp._loader import load_lib
        load_lib()
        return True
    except Exception:
        return False


def _comfy_configured() -> bool:
    return bool(os.environ.get("LOCALM_TEST_COMFY_URL"))


def _playwright_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("playwright") is not None


def _vulkan_split_configured() -> bool:
    """True once a second real Vulkan device is set up (e.g. Mesa lavapipe
    registered via VK_ADD_DRIVER_FILES) and its ICD manifest path is exported.
    Mirrors _comfy_configured()'s style deliberately: the gate only checks that
    the resource was set up, the actual "does the native ggml-vulkan backend
    really see 2 devices and split across them" assertion is the test body's
    job, not the gate's - see dev-notes/split-gpu-testing-research-2026-07-13.md
    Tier 1 and tests/test_gpu_split_native_vulkan.py."""
    return bool(os.environ.get("LOCALM_TEST_LAVAPIPE_ICD"))


def _real_multi_gpu_hardware_configured() -> bool:
    """True once opted into the Tier 2 real-hardware gate (any real 2-GPU box,
    owned or rented - lavapipe cannot approximate real VRAM pressure/allocator/
    OOM behavior or the amd-rocm/HIP backend at all). Same style as
    _vulkan_split_configured(): the gate only checks opt-in, the real
    assertions live in tests/test_gpu_split_real_hardware.py. See
    scripts/tier2_gpu_split/README.md.

    Deliberately says "owned or rented": this reason string is what a user
    READS on a skip, and saying "rented" described the environment someone
    imagined rather than the condition actually checked. The tests gate on two
    visible GPUs, an smi tool and a provisioned runtime - never on a rental -
    so a locally-owned 2-GPU box runs the whole gate for nothing."""
    return bool(os.environ.get("LOCALM_TEST_REAL_MULTI_GPU"))


_RESOURCE_GATES = (
    ("real_gguf", _runtime_available,
     "native llama runtime not provisioned (run 'localm setup-llama')"),
    ("real_comfy", _comfy_configured,
     "set LOCALM_TEST_COMFY_URL to a running ComfyUI"),
    ("real_browser", _playwright_available,
     "Playwright not installed (pip install playwright && playwright install)"),
    ("real_vulkan_split", _vulkan_split_configured,
     "set LOCALM_TEST_LAVAPIPE_ICD to a second Vulkan device's ICD manifest "
     "path (see dev-notes/split-gpu-testing-research-2026-07-13.md Tier 1)"),
    ("real_multi_gpu_hardware", _real_multi_gpu_hardware_configured,
     "set LOCALM_TEST_REAL_MULTI_GPU=1 on any real 2-GPU box, owned or rented "
     "(Tier 2 - see scripts/tier2_gpu_split/README.md)"),
)


_resource_available: dict = {}


def pytest_runtest_setup(item):
    """Skip resource-gated tests whose resource is unavailable - evaluated
    LAZILY, at a gated test's own setup, never at collection.

    This was an eager pytest_collection_modifyitems pass until 2026-07-22, and
    the eager form was a real bug: merely COLLECTING a real_gguf test (any run
    naming test_kv_bytes_offload.py, for example) ran _runtime_available() ->
    load_lib() in-process, mapping the bundled HIP/ROCm runtime into the shared
    pytest worker even when -m "not integration" deselected every gated test
    (pytest's own -m deselection hook is trylast, so it ran AFTER the gate).
    That cost seconds of native DLL load for tests that never run, and poisoned
    any LATER real `import torch` in the same process: torch's rocm_sdk preload
    of hipsolver.dll resolved its rocsolver.dll import BY NAME to the
    already-resident bundled build, which lacks the rocsolver_?sytrs_64
    entrypoints -> STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139), printed per
    affected test as a scary "Windows fatal exception" by pytest's
    faulthandler. Full root-cause evidence:
    dev-notes/pytest-collection-native-load-torch-fault-2026-07-22.md.

    Lazily, a deselected gated test triggers nothing, and a selected one loads
    the runtime at its own setup - exactly what it was about to do anyway.
    Results stay memoized per marker (and per xdist worker, as before, where
    each worker ran its own collection pass)."""
    for marker, check, reason in _RESOURCE_GATES:
        if marker not in item.keywords:
            continue
        ok = _resource_available.get(marker)
        if ok is None:
            ok = _resource_available[marker] = check()
        if not ok:
            pytest.skip(f"{marker}: {reason}")


@pytest.fixture(autouse=True)
def _isolate_localm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / ".localm"))


@pytest.fixture(autouse=True)
def _reset_comfy_readiness_cache():
    """comfy_client.py's ComfyUI readiness cache (_confirmed_alive) is a
    module-level set so it survives across requests within one real localm
    process - exactly the point of it - but that same persistence means it
    leaks between tests in the same pytest session: a test that confirms
    ComfyUI alive would let a LATER test's mocked-not-reachable case skip
    straight past _comfy_alive() via the cache and get a false "running"
    result. Clear it before and after every test so each starts cold."""
    from localm.media import comfy_client
    comfy_client._confirmed_alive.clear()
    yield
    comfy_client._confirmed_alive.clear()


@pytest.fixture(autouse=True)
def _clear_keep_diagnostics_env():
    """`localm gui --keep-diagnostics` sets LOCALM_KEEP_DIAGNOSTICS in-process; a
    test that exercises it would otherwise leak the env into later tests'
    keep_diagnostics_enabled() resolution. Clear it around every test."""
    os.environ.pop("LOCALM_KEEP_DIAGNOSTICS", None)
    yield
    os.environ.pop("LOCALM_KEEP_DIAGNOSTICS", None)


@pytest.fixture(autouse=True)
def _neutralise_bare_llama_pointers():
    """tests/_bare_llama.py's make_bare_llama() registers every instance it
    builds in a module-level list; a caller that overrides a pointer to a
    fake truthy value must have it nulled before that instance is garbage
    collected, or __del__ -> close() passes the fake address to the real
    native free. Runs after every test so no test file needs its own copy of
    this teardown."""
    yield
    from tests._bare_llama import neutralise_fake_pointers
    neutralise_fake_pointers()


@pytest.fixture(autouse=True)
def _reset_gpu_probe_cache():
    """discover.list_gpus() keeps a module-level last-known-good reading (served
    only when a probe overruns its deadline - there is deliberately NO TTL cache;
    every call re-probes). Without this, one test's mocked devices bleed into the
    next: a test that fakes two GPUs would leak them into a later "no GPU" test.

    Clearing alone is NOT sufficient, which is why _reset_gpu_probe_cache also
    bumps a probe epoch: an overrunning probe is abandoned rather than cancelled,
    so it outlives this fixture and writes its reading afterwards. A cold ROCm
    init (~6.5s) overruns the 4s deadline, so the real card landed in a LATER
    test that asserts a fake or empty reading. The epoch makes that late write a
    no-op. Runs before and after every test so each starts from a cold probe."""
    from localm import discover
    discover._reset_gpu_probe_cache()
    yield
    discover._reset_gpu_probe_cache()


@pytest.fixture(autouse=True)
def _neutralise_backend_vram_query():
    """loader.gpu_memory() reads the ACTIVE ggml backend's free VRAM (the signal
    GgufBackend._free_vram_bytes prefers). Once a real_gguf-gated test has RUN
    in this worker (its lazy resource gate above, or the test itself, calls
    load_lib() at that test's setup), _loaded_lib stays set for the rest of the
    session - which would make gpu_memory() return THIS machine's real free VRAM
    inside the many unit tests that simulate VRAM by patching
    _free_total_vram_bytes, silently defeating their mock. Force the resolver
    cache to the 'unavailable' sentinel so gpu_memory() returns None (and
    _free_vram_bytes falls back to the patched torch reader) unless a test opts
    in by setting the cache / patching gpu_memory itself.
    We do NOT reset _loaded_lib: dropping that reference could unload the DLL out
    from under an integration test's live model.

    _loader.native_lib_loaded() (added by #754) has the SAME loaded-runtime
    exposure in principle, but is deliberately NOT neutralised here (global,
    autouse, every test): tests/test_native_dll_conflict_guard.py directly unit-
    tests native_lib_loaded() itself by patching the _loaded_lib variable it reads
    - a blanket function-level override here would silently defeat that test's own
    mock instead of the real bug. See test_vram_preflight.py's own
    _neutralise_native_lib_loaded fixture (module-scoped, not global) for where
    this IS neutralised, for the specific tests that need it."""
    from localm.inference.backends.llamacpp import _loader
    saved = _loader._gpu_mem_cache
    _loader._gpu_mem_cache = False   # falsy, non-None -> gpu_memory() returns None
    yield
    _loader._gpu_mem_cache = saved


# --------------------------------------------------------------------------- #
#  No test may leave a giant file in tmp_path                                   #
#                                                                              #
#  truncate() is NOT sparse on Windows/NTFS: it allocates the blocks for real   #
#  (verified - `fsutil file queryValidData` on a 200 MB truncated file reports  #
#  Valid Data Length = the full 200,000,000; a sparse file reports 0). So a     #
#  test that truncates a fake multi-GB GGUF to drive a size-reading code path   #
#  writes those gigabytes for real, on every run, for every contributor.        #
#                                                                              #
#  This is ENFORCED rather than documented because documenting it demonstrably  #
#  did not work. It was fixed once in test_vram_eviction_safety.py (with the    #
#  reason written out in full) and warned about again in test_auto_gpu_layers   #
#  ("NEVER truncate() to GB sizes here") - and two other modules kept doing it  #
#  anyway, one commented "Sparse-ish: just truncate to size without writing     #
#  real bytes", the exact belief the earlier review had already disproved. It   #
#  ended up allocating ~315 GB per suite run and filling the disk to 99.5%      #
#  (#672). A third comment would not have caught the fourth violation.          #
# --------------------------------------------------------------------------- #

_MAX_TMP_FILE_BYTES = 100 * 1024 ** 2      # 100 MB


@pytest.fixture(autouse=True)
def _no_giant_tmp_files(tmp_path, request):
    """Fail a test that leaves a file over 100 MB in its tmp_path.

    Checks the OUTCOME (real bytes on disk) rather than the mechanism, because the
    mechanism cannot be intercepted: patching ``os.ftruncate`` does NOT catch
    ``fh.truncate()`` (verified - the C FileIO.truncate calls the syscall directly),
    and a static grep cannot see ``fh.truncate(size_bytes)`` where the size is a
    variable.

    Walks with ``os.walk`` + ``os.stat``, deliberately NOT ``Path.rglob`` /
    ``Path.stat``: this must measure the REAL bytes. Several tests legitimately
    monkeypatch ``Path.stat`` to report a huge st_size for a tiny file
    (test_vram_eviction_safety.py's _fake_stat_size), and that patch can still be
    live during this teardown - reading through it would fail exactly the tests
    doing the right thing. os.stat is not patched by those fakes, so it sees the
    truth.

    Drive-agnostic on purpose: it looks only at file SIZE, never at a path prefix,
    so it behaves identically wherever tmp_path lives (any drive, /tmp, CI).

    Never descends a link/junction, and never revisits a real directory. That is
    not paranoia: tests/test_rag_robustness_sweep.py deliberately builds BRANCHING
    self-referential junctions in its tmp_path (mklink /J loop1 -> .), and a
    Windows junction reports is_symlink() False, so a plain os.walk follows it and
    spins forever - the exact B3 DoS that localm/rag/store.py's _walk_files exists
    to avoid. A first draft of this guard used a naive os.walk and hung the suite
    at 92% on precisely that fixture.

    Opt out with ``@pytest.mark.allow_large_tmp_files`` when a test genuinely needs
    real bytes on disk (rare - prefer faking the size). Integration tests, which
    pull real models by design, are exempt automatically."""
    yield
    if request.node.get_closest_marker("allow_large_tmp_files"):
        return
    if request.node.get_closest_marker("integration"):
        return
    reparse = getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    big: list = []
    seen: set = set()
    try:
        for root, dirs, files in os.walk(tmp_path):
            # Prune links/junctions and any real dir already walked, IN PLACE, so
            # os.walk never recurses into a cycle (see the docstring).
            keep = []
            for d in dirs:
                dp = os.path.join(root, d)
                try:
                    st = os.lstat(dp)
                    if _stat.S_ISLNK(st.st_mode) or (
                            getattr(st, "st_file_attributes", 0) & reparse):
                        continue
                    rp = os.path.realpath(dp)
                    if rp in seen:
                        continue
                    seen.add(rp)
                except OSError:
                    continue
                keep.append(d)
            dirs[:] = keep
            for name in files:
                fp = os.path.join(root, name)
                try:
                    sz = os.stat(fp).st_size        # REAL size, never a faked one
                except OSError:
                    continue                        # vanished mid-walk; ignore
                if sz > _MAX_TMP_FILE_BYTES:
                    big.append((fp, sz))
    except OSError:
        return          # a test tearing its own tmp_path down; nothing to police
    if big:
        worst = "\n".join(f"    {sz / 1024**2:,.0f} MB  {os.path.relpath(p, tmp_path)}"
                          for p, sz in sorted(big, key=lambda t: -t[1])[:5])
        pytest.fail(
            f"{len(big)} file(s) over {_MAX_TMP_FILE_BYTES / 1024**2:.0f} MB left in "
            f"tmp_path:\n{worst}\n"
            "  truncate() is NOT sparse here - it writes those bytes for real, every\n"
            "  run, for every contributor (this once hit ~315 GB/run and filled the\n"
            "  disk). To drive a size-reading path, fake the size instead:\n"
            "      b._model_bytes = lambda: size_bytes\n"
            "  (see tests/test_auto_gpu_layers.py). If real bytes are genuinely\n"
            "  required, mark the test @pytest.mark.allow_large_tmp_files with a\n"
            "  why-comment.")


@pytest.fixture
def heavy_slot():
    """Let only ONE subprocess-heavy test run at a time, across every test file
    and every xdist worker.

    These tests start real Python interpreters, which is the whole point of them
    - but several landing on different workers at once measurably starves a
    neighbour: a test whose own lock budget then expires goes red for reasons
    that have nothing to do with its subject.

    SHARED here rather than defined per file, and that is the load-bearing part.
    It began as a private fixture in tests/test_rag_collection_lock.py; when
    tests/test_memory_cross_process_lock.py arrived with its own copy under a
    DIFFERENT slot filename, each file serialised its own heavy tests and the two
    files raced each other - reintroducing exactly the starvation the fixture
    exists to prevent, silently, because both copies looked correct in isolation.
    A lock whose identity is a string duplicated in two places is not one lock.
    Any new subprocess-heavy test should request this fixture.

    A plain O_EXCL file, because it has to work across PROCESSES (xdist workers
    are separate interpreters, so a threading lock would not be seen) and under a
    bare `-n auto` with no --dist loadgroup. The slot is force-taken if a crashed
    test ever leaves it behind, so this convenience lock can never wedge a run."""
    slot = Path(tempfile.gettempdir()) / "localm-subprocess-heavy.slot"
    deadline = time.time() + 240
    while True:
        try:
            os.close(os.open(str(slot), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if time.time() > deadline:
                break             # a leftover slot must never block the suite
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            slot.unlink()
        except OSError:
            pass


@pytest.fixture
def cli_runner(tmp_path, monkeypatch):
    """End-to-end CLI harness: a click CliRunner with a throwaway LOCALM_HOME.

    config.py freezes HOME_DIR / CONFIG_FILE / REGISTRY_FILE at import, so the
    autouse LOCALM_HOME env alone does not redirect load_config / save_config
    (which read the module attributes). Point them at the throwaway dir too so a
    CLI command that reads or writes config / registry never touches real data.
    """
    from click.testing import CliRunner
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return CliRunner()


def probe_double(reading, *, status=None):
    """Wrap a fake VRAM reading so it honours discover's OPT-IN ``return_status`` /
    ``deadline`` kwargs, for tests that patch ``discover.vram_info`` /
    ``discover.list_gpus`` / ``discover.vram_capacity``.

    The zero-arg doubles this replaces predate both kwargs, so a reader that opts in
    (switch_engine's pre-load gate, /v1/models/unload's reporting) hands them a kwarg
    they reject with TypeError. Swallowing that in production code was rejected: it
    would mask a genuine TypeError from inside a reader, and - worse - a status-blind
    double CANNOT express the probe_ok/timeout states the gate's behaviour now turns
    on, so the tests could not cover the very thing they exist to cover.

    ``status`` defaults to GPU_PROBE_OK because a patched reading IS the simulated
    result of a probe that COMPLETED - that is precisely what OK means, so this
    states a truth rather than assuming a convenient default. Pass
    ``status=GPU_PROBE_TIMEOUT`` / ``GPU_PROBE_BUSY`` to simulate a probe that did
    not complete, in which case ``reading`` is what the frozen last-known-good
    fallback would serve.

    ``reading`` may be a value or a zero-arg callable (for doubles that recompute
    per call, e.g. free VRAM that tracks which fake engines are currently loaded).
    ``deadline`` is accepted and ignored: these doubles are instant, so there is no
    cold init for a longer budget to wait out.
    """
    from localm.discover import GPU_PROBE_OK

    def _double(*args, **kwargs):
        value = reading() if callable(reading) else reading
        if kwargs.get("return_status"):
            return value, (status or GPU_PROBE_OK)
        return value

    return _double


def final_answer(result: str) -> str:
    """Strip the unconditional grounding footer (loop.py's Agent._grounding_footer)
    that run_task/chat/continue_task now append to every final answer, for tests
    that check the scripted text verbatim - see
    tests/plugins/coder/test_agent_loop_guards.py::TestGroundingFooter for the
    dedicated coverage of the footer itself.

    rfind, not find: the footer is always appended LAST, so the real footer is
    the LAST occurrence of the marker - find (first occurrence) would wrongly
    truncate a legitimate multi-paragraph answer that happens to mention this
    literal marker text earlier in its own prose. A missing footer (idx == -1)
    returns the string unchanged, which is deliberate: presence/absence of the
    footer is TestGroundingFooter's job, not this helper's."""
    marker = "\n\n[session record:"
    idx = result.rfind(marker)
    return result[:idx] if idx != -1 else result


def free_loopback_port() -> int:
    """A loopback port nothing is listening on, right now - for tests that need
    to prove a code path handles "unreachable" honestly, without depending on
    a well-known port (ComfyUI's 8188, this project's own 8642, ...) happening
    to be free on whoever's box runs the suite. A real install of any of those
    services on the test box is common, not hypothetical (see GitHub #955-
    class reports), and asserting behavior against "nothing is listening" must
    not silently become an assertion about "nothing ELSE is running today".

    Binds an ephemeral port and closes it immediately - never listened on, so
    it is free again by the time this returns, but nothing answers there
    either way. There is a theoretical reuse race (something else claims the
    exact same port in the gap before the caller dials it), astronomically
    unlikely in the ~16k-wide ephemeral range and the standard idiom for
    "find a free port" across the Python ecosystem."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()
