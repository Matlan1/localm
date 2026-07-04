# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared pytest fixtures.

Hermetic data dir: every test gets its own ``LOCALM_HOME`` under the test's
``tmp_path``. Without this, ``_detect_home()`` falls through to *portable mode*
(a ``home/`` directory next to the installed package) or the real ``~/.localm``
- so a developer who has run the GUI in portable mode would have the GUI tests
read, write, and DELETE their actual conversations/images while the suite runs.
``LOCALM_HOME`` takes priority over portable mode in ``_detect_home()``, so
pinning it here isolates every test. Tests that need a specific home override
this with their own ``monkeypatch.setenv`` (which runs after this autouse
fixture).
"""

import os
import tempfile
import shutil

# Isolate LOCALM_HOME globally at import time so that any module importing
# localm.config during test collection or execution resolves HOME_DIR to a
# temporary directory instead of the developer's real home config/keys.
_test_home_dir = tempfile.mkdtemp(prefix="localm_test_home_")
os.environ["LOCALM_HOME"] = _test_home_dir


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_test_home_dir, ignore_errors=True)


import pytest


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


_RESOURCE_GATES = (
    ("real_gguf", _runtime_available,
     "native llama runtime not provisioned (run 'localm setup-llama')"),
    ("real_comfy", _comfy_configured,
     "set LOCALM_TEST_COMFY_URL to a running ComfyUI"),
    ("real_browser", _playwright_available,
     "Playwright not installed (pip install playwright && playwright install)"),
)


def pytest_collection_modifyitems(config, items):
    available: dict = {}
    for item in items:
        for marker, check, reason in _RESOURCE_GATES:
            if marker not in item.keywords:
                continue
            ok = available.get(marker)
            if ok is None:
                ok = available[marker] = check()
            if not ok:
                item.add_marker(pytest.mark.skip(reason=f"{marker}: {reason}"))


@pytest.fixture(autouse=True)
def _isolate_localm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / ".localm"))


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
