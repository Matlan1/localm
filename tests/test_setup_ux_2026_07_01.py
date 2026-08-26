# SPDX-License-Identifier: AGPL-3.0-or-later
"""Setup/UX behaviour:

  - the console-script venv guard (localm / localcoder)
  - LOCALM_DEBUG=1 opens the debug log file
  - a bad LLAMA_CPP_LIB override warns instead of falling back silently
  - a broken (installed but unimportable-as-expected) localm_llama_runtime
    warns instead of falling back silently
"""

import logging
import sys
import types

import pytest


# --------------------------------------------------------------------------- #
#  The venv guard, and its console-script wiring
# --------------------------------------------------------------------------- #

def test_require_venv_exits_when_deps_absent(monkeypatch):
    # A bare interpreter: not a venv AND the runtime deps are not importable,
    # which is the one case the guard exits on.
    import localm._venvguard as vg
    monkeypatch.setattr(vg.sys, "prefix", "/usr")
    monkeypatch.setattr(vg.sys, "base_prefix", "/usr")     # prefix == base -> not a venv
    monkeypatch.setattr(vg, "_runtime_deps_present", lambda: False)
    with pytest.raises(SystemExit):
        vg.require_venv()


def test_require_venv_passes_inside_venv(monkeypatch):
    import localm._venvguard as vg
    monkeypatch.setattr(vg.sys, "prefix", "/proj/.venv")
    monkeypatch.setattr(vg.sys, "base_prefix", "/usr")     # differ -> a venv
    vg.require_venv()   # must not raise


def test_require_venv_passes_outside_venv_when_deps_present(monkeypatch):
    # A NON-.venv interpreter that DOES have the deps (pipx / container /
    # system-wide / cold install / CI editable) must run, not be blocked.
    import localm._venvguard as vg
    monkeypatch.setattr(vg.sys, "prefix", "/usr")
    monkeypatch.setattr(vg.sys, "base_prefix", "/usr")     # prefix == base -> not a venv
    monkeypatch.setattr(vg, "_runtime_deps_present", lambda: True)
    vg.require_venv()   # must not raise


def test_runtime_deps_present_true_in_this_env():
    # This test process has localm installed with its deps, so the probe is True -
    # i.e. the guard would let this interpreter run even outside a .venv.
    import localm._venvguard as vg
    assert vg._runtime_deps_present() is True


def test_console_main_entrypoints_import():
    # The pyproject [project.scripts] targets must resolve to callables.
    from localm.cli import console_main as localm_entry
    from localm.plugins.coder.cli import console_main as coder_entry
    assert callable(localm_entry) and callable(coder_entry)


def test_bare_main_still_importable_for_in_process_callers():
    # The `localm coder` route and the test suite call `main` directly, so it
    # stays importable and separate from the guarded console_main.
    from localm.cli import main as group
    from localm.plugins.coder.cli import main as coder_cmd
    assert group is not None and coder_cmd is not None


# --------------------------------------------------------------------------- #
#  LOCALM_DEBUG=1 opens a real log file
# --------------------------------------------------------------------------- #

def test_honor_env_debug_opens_log_on_env_request(tmp_path, monkeypatch):
    from localm import debuglog
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setenv("LOCALM_DEBUG", "1")                # a request, not a path
    saved = list(debuglog.logger.handlers)
    try:
        assert debuglog.log_file_path() is None            # "1" is not a log path
        debuglog.honor_env_debug()
        p = debuglog.log_file_path()
        assert p is not None and p.exists(), "LOCALM_DEBUG=1 must write a log file"
    finally:
        for h in list(debuglog.logger.handlers):
            if h not in saved:
                debuglog.logger.removeHandler(h)


def test_honor_env_debug_noop_when_unset(monkeypatch):
    from localm import debuglog
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    before = list(debuglog.logger.handlers)
    debuglog.honor_env_debug()
    assert list(debuglog.logger.handlers) == before        # nothing opened


# --------------------------------------------------------------------------- #
#  A bad explicit LLAMA_CPP_LIB warns
# --------------------------------------------------------------------------- #

def test_bad_llama_cpp_lib_warns(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "_warned_explicit_lib", False)
    monkeypatch.setenv("LLAMA_CPP_LIB", str(tmp_path / "nope" / "llama.dll"))

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Capture()
    _loader.logger.addHandler(h)
    try:
        _loader.runtime_binary_dir()   # resolves to bundled runtime or None, not the bad path
    finally:
        _loader.logger.removeHandler(h)

    assert any("LLAMA_CPP_LIB" in m for m in records), \
        "a bad LLAMA_CPP_LIB override must surface a warning, not silently fall back"


def test_bad_llama_cpp_lib_warns_once(tmp_path, monkeypatch):
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "_warned_explicit_lib", False)
    monkeypatch.setenv("LLAMA_CPP_LIB", str(tmp_path / "nope" / "llama.dll"))

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Capture()
    _loader.logger.addHandler(h)
    try:
        _loader.runtime_binary_dir()
        _loader.runtime_binary_dir()
        _loader.runtime_binary_dir()
    finally:
        _loader.logger.removeHandler(h)

    hits = [m for m in records if "LLAMA_CPP_LIB" in m]
    assert len(hits) == 1, f"the warning must fire once per process, got {len(hits)}"


# --------------------------------------------------------------------------- #
#  A broken localm_llama_runtime install warns, but a simply-not-installed one
#  (the normal pre-`setup-llama` state) stays silent
# --------------------------------------------------------------------------- #

def test_broken_llama_runtime_warns(monkeypatch):
    from localm.inference.backends.llamacpp import _loader

    fake = types.ModuleType("localm_llama_runtime")   # no lib_dir() attribute
    monkeypatch.setitem(sys.modules, "localm_llama_runtime", fake)
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Capture()
    _loader.logger.addHandler(h)
    try:
        _loader.runtime_binary_dir()
    finally:
        _loader.logger.removeHandler(h)

    assert any("localm_llama_runtime" in m and "broken" in m for m in records), \
        "an installed-but-broken localm_llama_runtime must warn, not silently fall back"


def test_missing_llama_runtime_stays_silent(monkeypatch):
    from localm.inference.backends.llamacpp import _loader

    monkeypatch.setitem(sys.modules, "localm_llama_runtime", None)  # simulates ImportError
    monkeypatch.delenv("LLAMA_CPP_LIB", raising=False)

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Capture()
    _loader.logger.addHandler(h)
    try:
        _loader.runtime_binary_dir()
    finally:
        _loader.logger.removeHandler(h)

    assert records == [], \
        "a simply-not-yet-installed runtime wheel is the normal pre-setup state, not a warning"
