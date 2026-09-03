# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pytest fixtures.

Hermetic data dir: every test gets its own ``LOCALM_HOME`` under the test's
``tmp_path``. ``LOCALM_HOME`` takes priority over portable mode in
``_detect_home()``, so pinning it here isolates every test. Tests that need a
specific home override this with their own ``monkeypatch.setenv`` (which runs
after this autouse fixture).

That import-time directory is removed at PROCESS EXIT, not only at the end of a
pytest session. The removal is conditional; see the temp-root section below.
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

# Isolate LOCALM_HOME at import time so that any module importing localm.config
# during collection or execution resolves HOME_DIR to a temporary directory.
_test_home_dir = tempfile.mkdtemp(prefix="localm_test_home_")
os.environ["LOCALM_HOME"] = _test_home_dir

# Disable the media-plugin legacy-workflow migration for the whole suite. The
# flag is inherited by spawned subprocesses.
os.environ["LOCALM_SKIP_LEGACY_WORKFLOW_MIGRATION"] = "1"


def pytest_sessionfinish(session, exitstatus):
    _cleanup_test_home()
    _report_system_path_touches(session)
    _report_installer_runs(session)
    _report_wrong_temp_root(session)


import pytest


# --------------------------------------------------------------------------- #
#  The import-time temp home is cleaned up at PROCESS EXIT, unless it landed    #
#  on a temp root the run did not intend: that directory is kept and the run    #
#  goes red naming it.                                                          #
#                                                                              #
#  The intended root reaches an xdist worker over execnet's workerinput         #
#  channel, not the environment.                                                #
# --------------------------------------------------------------------------- #

_EXPECTED_TEMP_ROOT_KEY = "localm_expected_temp_root"
_EXPECTED_TEMP_ROOT_ENV = "LOCALM_TEST_EXPECTED_TEMP_ROOT"

# Where this process's temp root actually resolved, read off the directory
# mkdtemp just made rather than from a second gettempdir() call.
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


def make_console_wide_and_plain(monkeypatch, width: str = "300") -> None:
    """Make every Rich console in this test render wide and uncolored,
    matching CI's real terminal (a non-tty that still reports
    ``is_terminal`` True, so Console.size short-circuits to 80x25 before it
    ever reads COLUMNS, and Console.__init__ separately caches a color
    system at construction that later env changes cannot revisit).

    Two independent things, both needed:

    - ``is_dumb_terminal`` is read live on every ``.size`` access, so a
      class-level patch fixes every Console - the shared CLI singleton and
      any instance a command constructs fresh per call.
    - ``_color_system`` is computed ONCE in ``__init__`` and cached on the
      instance forever. A class-level patch to ``_detect_color_system``
      only helps instances built AFTER the patch (a command's own
      per-call ``Console()``); the shared singletons imported once per
      xdist worker were already constructed, possibly under a DIFFERENT
      test's environment in the same worker process, so their cached value
      is forced directly.

    See test_wide_dumb_console_helper_actually_works for the two-arm proof.
    """
    import rich.console
    monkeypatch.setattr(rich.console.Console, "is_dumb_terminal", False)
    monkeypatch.setattr(rich.console.Console, "_detect_color_system",
                        lambda self: None)
    from localm.cli import _core as _core_mod
    monkeypatch.setattr(_core_mod.console, "_color_system", None)
    from localm.model_manager import _shared as _shared_mod
    monkeypatch.setattr(_shared_mod.console, "_color_system", None)
    monkeypatch.setenv("COLUMNS", width)


def _same_dir(a, b) -> bool:
    """True when two paths name the same directory.

    Normalised strings first, which is the entire answer whenever both sides
    were resolved from the same TEMP value. Only when THAT disagrees is realpath
    consulted, so a mere difference in spelling (an 8.3 short name, a junction)
    is not reported as a misplaced run. The realpath call touches the
    filesystem."""
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
    # A controller-stamped value always wins, so the env var can only add the
    # check to a run that declared none.
    from_env = os.environ.get(_EXPECTED_TEMP_ROOT_ENV) or None
    if workerinput is not None:
        _expected_temp_root = workerinput.get(_EXPECTED_TEMP_ROOT_KEY) or from_env
        if _expected_temp_root is None:
            # A worker that was never told which root the run intended: the
            # check below cannot arm, so report that instead of cleaning up.
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
        channel rather than the environment.

        The controller's OWN resolved root is the fallback expectation. A
        declared root wins over it."""
        node.workerinput[_EXPECTED_TEMP_ROOT_KEY] = (
            _expected_temp_root or _actual_temp_root)


def _cleanup_test_home():
    """Remove the import-time LOCALM_HOME, unless it is evidence.

    Idempotent: pytest_sessionfinish calls it at the end of a session and atexit
    catches every other way a process can end, including having no session at
    all."""
    if _wrong_temp_root:
        return                  # KEEP it - this directory IS the evidence
    try:
        shutil.rmtree(_test_home_dir)
    except FileNotFoundError:
        pass                    # already gone: sessionfinish ran, then atexit
    except OSError as exc:
        # Report a directory that could not be removed, on the ORIGINAL stderr
        # and guarded: this also runs from atexit, where pytest's capture
        # replacement may already be torn down.
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

    Once per PROCESS, not once per test: under -n auto every test in a misplaced
    worker would otherwise fail.

    A fixture as well as pytest_sessionfinish, because a worker's exitstatus
    does not become the run's under -n auto but a failed test does."""
    global _wrong_temp_root_reported
    if _wrong_temp_root and not _wrong_temp_root_reported:
        _wrong_temp_root_reported = True
        pytest.fail(_wrong_temp_root + "\n" + _WRONG_TEMP_ROOT_ADVICE)


def _report_wrong_temp_root(session):
    """Session-level backstop for the process the fixture cannot reach.

    Under -n auto the CONTROLLER runs no tests at all, so the per-test fixture
    never fires there. The controller's exitstatus IS the run's, so failing here
    covers that case."""
    if not _wrong_temp_root:
        return
    print("\nWRONG TEMP ROOT:\n  " + _wrong_temp_root + "\n"
          + _WRONG_TEMP_ROOT_ADVICE)
    if getattr(session, "exitstatus", 0) == 0:
        session.exitstatus = 1


# --------------------------------------------------------------------------- #
#  No test may touch a real system path                                         #
#                                                                              #
#  Marker roots are DERIVED, never a hardcoded drive literal: the Windows       #
#  system root comes from %SystemRoot%/%windir% at runtime. The guard only      #
#  ever compares STRINGS - it never opens, stats, or reads any system path.     #
#                                                                              #
#  Both `builtins.open` AND `io.open` are patched: bare open() goes through     #
#  the first, Path.read_text()/Path.open() through the second.                  #
#                                                                              #
#  /proc, /sys, /dev and /usr are not marked. The markers are config and        #
#  credential roots.                                                            #
# --------------------------------------------------------------------------- #

_SYSPATH_EXTRA_ENV = "LOCALM_TEST_SYSPATH_EXTRA_MARKERS"

# (kind, path, origin frame) -> count.
_SYSPATH_HITS: dict = {}
_SYSPATH_ARMED = False
_GUARD_FILE = __file__.replace("\\", "/")


def _syspath_marker_roots() -> list:
    """The path prefixes that count as a real system location, lowercased and
    slash-normalised.

    ``_SYSPATH_EXTRA_ENV`` ADDS roots and can never remove one."""
    roots = []
    for var in ("SystemRoot", "windir"):
        value = os.environ.get(var)
        if value:
            roots.append(value)
    if os.name != "nt":
        # FHS config/credential roots; the runtime interfaces (/proc, /sys,
        # /dev) are not marked.
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


# ssl.create_default_context() (localm/http_ssl.py's verified_urlopen, using
# the platform's native store by design) reads a small, fixed set of PUBLIC
# trust-anchor paths under /etc on POSIX - real filesystem access, but
# categorically different from the credentials/config /etc otherwise marks:
# a CA bundle is data every TLS client on the machine needs to read. See
# test_conftest_syspath_guard.py's TestTheGuardExemptsTheRealCertStoreProbe
# for the exact set this was measured against and the negative case proving
# the exemption stays narrow.
_SYSPATH_CERT_STORE_ALLOW = frozenset({
    "/etc/ssl/certs",
    "/etc/ssl/cert.pem",
    "/etc/pki/tls/cert.pem",
    "/etc/pki/tls/certs",
    "/etc/ssl/ca-bundle.pem",
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
})


def _is_cert_store_probe(raw) -> bool:
    try:
        s = os.fspath(raw)
    except TypeError:
        return False
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    elif not isinstance(s, str):
        return False
    return s.replace("\\", "/").rstrip("/").lower() in _SYSPATH_CERT_STORE_ALLOW


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
    walking the stack for the originating frame runs ONLY after a marker has
    already matched."""
    global _SYSPATH_ARMED
    if _SYSPATH_ARMED:
        return True
    rx = _syspath_regex(_syspath_marker_roots())
    if rx is None:
        return False

    def _wrap(kind, func):
        def guarded(path, *args, **kwargs):
            if _syspath_matches(rx, path) and not _is_cert_store_probe(path):
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


# Warm the stdlib mimetypes registry before arming the guard: the first
# guess_type() consults the OS mime registry, which on POSIX reads
# /etc/mime.types plus the httpd/apache paths. guess_type(), not init(), so it
# goes through the stdlib's own lazy init and warms exactly once per process.
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

    Enforced per-test, not only at session end: under ``-n auto`` a worker's
    session exitstatus does not become the run's, but a failed test does. The
    session report below still runs, for anything recorded outside a test
    (import or collection time)."""
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
    its own, so this is a no-op there."""
    if not _SYSPATH_HITS:
        return
    print("\nSYSTEM PATH TOUCHES DETECTED (tests must stay inside tmp_path):\n"
          + _format_syspath_hits(_SYSPATH_HITS) + "\n" + _SYSPATH_ADVICE)
    if getattr(session, "exitstatus", 0) == 0:
        session.exitstatus = 1


# --------------------------------------------------------------------------- #
#  No test may install packages INTO THE INTERPRETER RUNNING THE SUITE         #
#                                                                              #
#  Installing into a DISPOSABLE venv under tmp_path stays allowed: the line is #
#  the install TARGET, not the act.                                            #
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


# pip, pip3, pip3.12 - not pipx.
_PIP_SHIM = re.compile(r"^pip[0-9.]*$")


def _installs_into_this_interpreter(cmd):
    """The reason string when *cmd* installs into the running interpreter, else None.

    Narrow. It does not fire on `pip cache dir`, `pip freeze`, `pip list` or
    `pip --version` (a real pip child that installs NOTHING), nor on an install
    aimed at ANOTHER interpreter - installing into a disposable venv under
    tmp_path is legitimate.

    A bare `pip install x` has no directory component, so resolving it as a path
    would land on the CWD and match neither sys.executable nor sys.prefix. A
    bare `pip`/`uv` is found on PATH, and under the suite PATH leads to the venv
    running it, so no explicit target is read as THIS interpreter."""
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
    replaces this wrapper for that test and is never flagged."""
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
                # A spawn from a background thread may land while a different
                # test is current, so the nodeid can name the wrong test. Say so
                # and print the call path.
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
                # Block rather than record. The offending test fails here and
                # is named by the fixture below.
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
    """Fail the test that spawned it. Enforced per-test like the sibling guard:
    under ``-n auto`` a worker's exitstatus does not become the run's, but a
    failed test does."""
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
#  A test tagged real_gguf / real_comfy / real_browser is skipped, never       #
#  failed, unless its resource is actually available.                          #
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
    The gate only checks that the resource was set up; whether the native
    ggml-vulkan backend really sees 2 devices and splits across them is the test
    body's assertion."""
    return bool(os.environ.get("LOCALM_TEST_LAVAPIPE_ICD"))


def _real_multi_gpu_hardware_configured() -> bool:
    """True once opted into the Tier 2 real-hardware gate (any real 2-GPU box,
    owned or rented). The gate only checks opt-in; the real assertions live in
    the Tier 2 tests, which gate on two visible GPUs, an smi tool and a
    provisioned runtime - never on a rental."""
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
     "path"),
    ("real_multi_gpu_hardware", _real_multi_gpu_hardware_configured,
     "set LOCALM_TEST_REAL_MULTI_GPU=1 on any real 2-GPU box, owned or rented "
     "(Tier 2 - see scripts/tier2_gpu_split/README.md)"),
)


_resource_available: dict = {}


def pytest_runtest_setup(item):
    """Skip resource-gated tests whose resource is unavailable - evaluated
    LAZILY, at a gated test's own setup, never at collection.

    A deselected gated test triggers nothing, and a selected one loads the
    runtime at its own setup. Results are memoized per marker, per process."""
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
    module-level set, so it persists across tests in one pytest session. Clear
    it before and after every test so each starts cold."""
    from localm.media import comfy_client
    comfy_client._confirmed_alive.clear()
    yield
    comfy_client._confirmed_alive.clear()


@pytest.fixture(autouse=True)
def _clear_keep_diagnostics_env():
    """`localm gui --keep-diagnostics` sets LOCALM_KEEP_DIAGNOSTICS in-process.
    Clear it around every test."""
    os.environ.pop("LOCALM_KEEP_DIAGNOSTICS", None)
    yield
    os.environ.pop("LOCALM_KEEP_DIAGNOSTICS", None)


@pytest.fixture(autouse=True)
def _reset_coder_privacy_registry():
    """A top-level coder Agent publishes its privacy mode into a process-global
    counter in ``__init__`` and only takes it back out in ``close()``. Agent's
    default mode IS privacy, and test files build agents and let them fall out
    of scope without closing them, so the count carries into every later test in
    the same worker.

    What that breaks is not the coder: ``debuglog.debug_content_enabled()``
    consults the same counter and suppresses raw chat content whenever any coder
    session is in privacy mode. So one unclosed agent silently turns off raw
    content logging for the rest of the worker, and a test asserting that the
    debug log captured raw model output fails while passing on its own. Restores
    the count rather than zeroing it, so a test that deliberately registers one
    and checks the gate still sees its own effect.
    """
    import localm.audit as audit
    before = audit._active_coder_privacy_count
    yield
    audit._active_coder_privacy_count = before


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
    """discover.list_gpus() keeps a module-level last-known-good reading, served
    only when a probe overruns its deadline; there is no TTL cache and every
    call re-probes.

    ``discover._reset_gpu_probe_cache()`` clears that reading AND bumps a probe
    epoch: an overrunning probe is abandoned rather than cancelled, so it can
    outlive this fixture and write its reading afterwards, and the epoch makes
    that late write a no-op. Runs before and after every test so each starts
    from a cold probe."""
    from localm import discover
    discover._reset_gpu_probe_cache()
    yield
    discover._reset_gpu_probe_cache()


@pytest.fixture(autouse=True)
def _neutralise_backend_vram_query():
    """loader.gpu_memory() reads the ACTIVE ggml backend's free VRAM (the signal
    GgufBackend._free_vram_bytes prefers). Once a real_gguf-gated test has RUN
    in this worker, _loaded_lib stays set for the rest of the session and
    gpu_memory() would return this machine's real free VRAM inside unit tests
    that simulate VRAM by patching _free_total_vram_bytes. Forces the resolver
    cache to the 'unavailable' sentinel so gpu_memory() returns None, and
    _free_vram_bytes falls back to the patched torch reader, unless a test opts
    in by setting the cache or patching gpu_memory itself.

    Does NOT reset _loaded_lib: dropping that reference could unload the DLL out
    from under an integration test's live model.

    Does NOT neutralise _loader.native_lib_loaded() either: it is unit-tested
    directly by patching the _loaded_lib variable it reads, so a global override
    here would defeat that test's own mock. test_vram_preflight.py neutralises
    it module-scoped for the tests that need it."""
    from localm.inference.backends.llamacpp import _loader
    saved = _loader._gpu_mem_cache
    _loader._gpu_mem_cache = False   # falsy, non-None -> gpu_memory() returns None
    yield
    _loader._gpu_mem_cache = saved


# --------------------------------------------------------------------------- #
#  No test may leave a giant file in tmp_path                                   #
# --------------------------------------------------------------------------- #

_MAX_TMP_FILE_BYTES = 100 * 1024 ** 2      # 100 MB


@pytest.fixture(autouse=True)
def _no_giant_tmp_files(tmp_path, request):
    """Fail a test that leaves a file over 100 MB in its tmp_path.

    Checks the OUTCOME (real bytes on disk), not the mechanism: patching
    ``os.ftruncate`` does not catch ``fh.truncate()``, and a static grep cannot
    see ``fh.truncate(size_bytes)`` where the size is a variable.

    Walks with ``os.walk`` + ``os.stat``, never ``Path.rglob`` / ``Path.stat``:
    tests monkeypatch ``Path.stat`` to report a huge st_size for a tiny file,
    and that patch can still be live during this teardown. os.stat is not
    patched by those fakes.

    Looks only at file SIZE, never at a path prefix, so it behaves identically
    wherever tmp_path lives.

    Never descends a link or junction, and never revisits a real directory: a
    Windows junction reports is_symlink() False, so a plain os.walk follows a
    self-referential one and spins forever.

    Opt out with ``@pytest.mark.allow_large_tmp_files`` when a test genuinely
    needs real bytes on disk. Integration tests are exempt automatically."""
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
            # Prune links/junctions and any real dir already walked, in place,
            # so os.walk never recurses into a cycle.
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

    SHARED here, never copied per file: one slot filename is what makes it one
    lock. Any new subprocess-heavy test should request this fixture.

    A plain O_EXCL file, so it works across PROCESSES (xdist workers are
    separate interpreters, so a threading lock would not be seen) and under a
    bare `-n auto` with no --dist loadgroup. The slot is force-taken after the
    deadline if a crashed test leaves it behind, so it can never wedge a run."""
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

    ``status`` defaults to GPU_PROBE_OK: a patched reading is the simulated
    result of a probe that COMPLETED. Pass ``status=GPU_PROBE_TIMEOUT`` /
    ``GPU_PROBE_BUSY`` to simulate a probe that did not complete, in which case
    ``reading`` is what the frozen last-known-good fallback would serve.

    ``reading`` may be a value or a zero-arg callable (for doubles that recompute
    per call, e.g. free VRAM that tracks which fake engines are currently loaded).
    ``deadline`` is accepted and ignored: these doubles are instant.
    """
    from localm.discover import GPU_PROBE_OK

    def _double(*args, **kwargs):
        value = reading() if callable(reading) else reading
        if kwargs.get("return_status"):
            return value, (status or GPU_PROBE_OK)
        return value

    return _double


def final_answer(result: str) -> str:
    """Strip the unconditional grounding footer (loop.py's
    Agent._grounding_footer) that run_task/chat/continue_task append to every
    final answer, for tests that check the scripted text verbatim.

    rfind, not find: the footer is always appended LAST, so the real footer is
    the LAST occurrence of the marker. A missing footer (idx == -1) returns the
    string unchanged."""
    marker = "\n\n[session record:"
    idx = result.rfind(marker)
    return result[:idx] if idx != -1 else result


def free_loopback_port() -> int:
    """A loopback port nothing is listening on, right now - for tests that need
    to prove a code path handles "unreachable" honestly, without depending on a
    well-known port (ComfyUI's 8188, this project's own 8642) being free on
    whoever's box runs the suite.

    Binds an ephemeral port and closes it immediately. It is never listened on,
    so it is free again by the time this returns. There is a theoretical reuse
    race if something else claims the same port before the caller dials it."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()
