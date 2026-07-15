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


def _vulkan_split_configured() -> bool:
    """True once a second real Vulkan device is set up (e.g. Mesa lavapipe
    registered via VK_ADD_DRIVER_FILES) and its ICD manifest path is exported.
    Mirrors _comfy_configured()'s style deliberately: the gate only checks that
    the resource was set up, the actual "does the native ggml-vulkan backend
    really see 2 devices and split across them" assertion is the test body's
    job, not the gate's - see dev-notes/split-gpu-testing-research-2026-07-13.md
    Tier 1 and tests/test_gpu_split_native_vulkan.py."""
    return bool(os.environ.get("LOCALM_TEST_LAVAPIPE_ICD"))


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
    GgufBackend._free_vram_bytes prefers). The real_gguf resource gate above calls
    load_lib() at COLLECTION time, so _loaded_lib is set for the whole session -
    which would make gpu_memory() return THIS machine's real free VRAM inside the
    many unit tests that simulate VRAM by patching _free_total_vram_bytes, silently
    defeating their mock. Force the resolver cache to the 'unavailable' sentinel so
    gpu_memory() returns None (and _free_vram_bytes falls back to the patched torch
    reader) unless a test opts in by setting the cache / patching gpu_memory itself.
    We do NOT reset _loaded_lib: dropping that reference could unload the DLL out
    from under an integration test's live model."""
    from localm.inference.backends.llamacpp import _loader
    saved = _loader._gpu_mem_cache
    _loader._gpu_mem_cache = False   # falsy, non-None -> gpu_memory() returns None
    yield
    _loader._gpu_mem_cache = saved


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
