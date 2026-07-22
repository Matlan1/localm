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
"""

import os
import stat as _stat
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
    so it behaves identically wherever tmp_path lives (C:, D:, /tmp, CI).

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
