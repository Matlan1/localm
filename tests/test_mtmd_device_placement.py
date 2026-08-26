# SPDX-License-Identifier: AGPL-3.0-or-later
"""The vision projector must not sit on a device the user's configuration excluded.

Field evidence (issue 1222): ``clip_ctx: CLIP using Vulkan0 backend`` while
``auto GPU split: device 1: 50%, device 2: 50%`` - the projector held ~857 MiB of
weights plus a 248 MiB compute buffer on the one card the configured split left out.
``mtmd_context_params`` has no device field; clip reads the process environment
variable ``MTMD_BACKEND_DEVICE`` instead (upstream ``tools/mtmd/clip.cpp:184-195``).

Two properties are pinned here, and they pull in opposite directions on purpose:

* the variable IS set, to the right name, when localm can determine it exactly;
* it is NOT set when localm cannot - because ggml's registry order and llama.cpp's
  ``model->devices`` order are NOT the same sequence (llama.cpp drops iGPUs whenever
  a discrete GPU exists, skips ACCEL, dedups by device_id and hoists RPC), so a
  positional lookup is only valid when every non-CPU device is a plain GPU.

The second half is the one worth having: a wrong name is non-fatal upstream
(clip warns and falls back), so nothing here would CRASH if the guard were dropped -
it would just quietly move a gigabyte onto an arbitrary card. See
dev-notes/mmproj-device-placement-2026-08-12.md.
"""

import os

import pytest

from localm.inference.backends.llamacpp import mtmd as mtmd_mod

CPU = 0      # GGML_BACKEND_DEVICE_TYPE_CPU
GPU = 1      # GGML_BACKEND_DEVICE_TYPE_GPU
IGPU = 2     # GGML_BACKEND_DEVICE_TYPE_IGPU  (b8100+; ACCEL moved to 3)
ACCEL = 3    # GGML_BACKEND_DEVICE_TYPE_ACCEL (b8100+)


@pytest.fixture(autouse=True)
def _no_inherited_env(monkeypatch):
    """MTMD_BACKEND_DEVICE is process-global and these tests assert on its exact
    presence/absence, so start every one of them from a known-unset state. A value
    inherited from the developer's own shell would otherwise make the
    "defers to the user" test pass vacuously."""
    monkeypatch.delenv(mtmd_mod._MTMD_DEVICE_ENV, raising=False)


def _devices(monkeypatch, devices):
    """Patch the ggml device registry read that the resolver consumes.

    Patches ``compute_devices`` on the _loader MODULE, which is what the resolver
    looks up at call time - not a copy imported into mtmd's namespace, which would
    leave the real (library-loading) function running."""
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "compute_devices", lambda: list(devices))


class TestResolveBackendDeviceName:
    """The guard: resolve a name only when the index space is provably positional."""

    def test_index_zero_resolves_to_nothing(self, monkeypatch):
        """Device 0 is already clip's own default, so there is nothing to correct.

        Returning None here (rather than the correct name for device 0) is what
        keeps every default install byte-identical."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        assert mtmd_mod._resolve_backend_device_name(0) is None

    def test_negative_index_resolves_to_nothing(self, monkeypatch):
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        assert mtmd_mod._resolve_backend_device_name(-1) is None

    def test_all_discrete_gpus_resolves_by_position(self, monkeypatch):
        """THE FIX. Every non-CPU device is a plain GPU, so llama.cpp's
        model->devices is exactly this sequence and index 1 IS Vulkan1."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        assert mtmd_mod._resolve_backend_device_name(1) == "Vulkan1"

    def test_cpu_device_position_does_not_shift_the_index(self, monkeypatch):
        """The CPU device is skipped wherever it sits in the registry. Ordering it
        LAST here is deliberate: a resolver that sliced ``devices[1:]`` instead of
        filtering by type would pass the test above and fail this one."""
        _devices(monkeypatch, [("Vulkan0", GPU), ("Vulkan1", GPU), ("CPU", CPU)])
        assert mtmd_mod._resolve_backend_device_name(1) == "Vulkan1"

    def test_third_device_resolves(self, monkeypatch):
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU),
                               ("Vulkan1", GPU), ("Vulkan2", GPU)])
        assert mtmd_mod._resolve_backend_device_name(2) == "Vulkan2"

    def test_integrated_gpu_present_refuses(self, monkeypatch):
        """THE DIVERGENCE THAT ACTUALLY BITES, and the reason this guard exists.

        ggml-vulkan enumerates integrated GPUs and types them IGPU
        (ggml-vulkan.cpp:17878); llama.cpp's llama_prepare_model_devices admits an
        iGPU ONLY when no discrete GPU was found. So on this registry llama.cpp's
        device list is ["Vulkan0"] alone - one entry - and index 1 does not name
        Vulkan1 there, it names nothing at all. Refuse."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", IGPU)])
        assert mtmd_mod._resolve_backend_device_name(1) is None

    def test_integrated_gpu_first_also_refuses(self, monkeypatch):
        """Same divergence, opposite direction: with the iGPU enumerated FIRST,
        llama.cpp's index 0 is the DISCRETE card while this list's index 0 is the
        integrated one. Every index is wrong, not merely the last."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", IGPU), ("Vulkan1", GPU)])
        assert mtmd_mod._resolve_backend_device_name(1) is None

    def test_accel_device_present_refuses(self, monkeypatch):
        """llama.cpp skips ACCEL entirely when building model->devices, so an
        ACCEL entry ahead of a GPU shifts every later index by one."""
        _devices(monkeypatch, [("CPU", CPU), ("BLAS", ACCEL),
                               ("Vulkan0", GPU), ("Vulkan1", GPU)])
        assert mtmd_mod._resolve_backend_device_name(1) is None

    def test_index_out_of_range_refuses(self, monkeypatch):
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU)])
        assert mtmd_mod._resolve_backend_device_name(1) is None

    def test_no_gpu_devices_refuses(self, monkeypatch):
        _devices(monkeypatch, [("CPU", CPU)])
        assert mtmd_mod._resolve_backend_device_name(1) is None

    def test_empty_device_name_refuses(self, monkeypatch):
        """An unnamed device cannot be handed to MTMD_BACKEND_DEVICE at all;
        setting it to "" would read as unset to clip anyway."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("", GPU)])
        assert mtmd_mod._resolve_backend_device_name(1) is None

    def test_registry_read_failure_refuses(self, monkeypatch):
        """A probe failure must lose the pinning, never the vision."""
        from localm.inference.backends.llamacpp import _loader

        def _boom():
            raise OSError("registry unavailable")

        monkeypatch.setattr(_loader, "compute_devices", _boom)
        assert mtmd_mod._resolve_backend_device_name(1) is None

    def test_refusal_is_logged_with_its_reason(self, monkeypatch, caplog):
        """Rule 5: the user asked for device 1 and the projector is NOT going
        there. That decision has to reach the always-on ring buffer (INFO+), not
        vanish."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", IGPU)])
        with caplog.at_level("INFO", logger="localm"):
            assert mtmd_mod._resolve_backend_device_name(1) is None
        assert any("mixes device types" in r.getMessage() for r in caplog.records), \
            f"no reason logged; records were {[r.getMessage() for r in caplog.records]}"


class _FakeLib:
    """Just enough of the mtmd CDLL to drive the REAL _open().

    Records the value of MTMD_BACKEND_DEVICE *as seen from inside* the native call,
    which is the only moment that matters: the variable is process-global and is
    unset again the instant _open returns."""

    def __init__(self, *, fail=False, ret=0x1234):
        self.seen_env = "<never called>"
        self.calls = 0
        self._fail = fail
        self._ret = ret

    def mtmd_context_params_default(self):
        return mtmd_mod._MtmdParams()

    def mtmd_init_from_file(self, path, model_ptr, params):
        self.calls += 1
        self.seen_env = os.environ.get(mtmd_mod._MTMD_DEVICE_ENV)
        if self._fail:
            raise RuntimeError("native init blew up")
        return self._ret


def _ctx(lib, gpu_index):
    """A real MtmdContext with its native-loading __init__ bypassed (the same
    convention test_mtmd_vision.py uses), so _open runs for real."""
    ctx = mtmd_mod.MtmdContext.__new__(mtmd_mod.MtmdContext)
    ctx._m = lib
    ctx._mmproj_path = "/fake/mmproj.gguf"
    ctx._model_ptr = 0xBEEF
    ctx._gpu_index = gpu_index
    return ctx


class TestOpenSetsTheEnvironment:

    def test_env_is_visible_to_the_native_call_and_gone_afterwards(self, monkeypatch):
        """THE PROPERTY. Assert on what the native call SAW, not on the return
        value: a variable set after the call, or never set at all, produces the
        identical return code."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        lib = _FakeLib()
        assert _ctx(lib, 1)._open(use_gpu=True) == 0x1234
        assert lib.seen_env == "Vulkan1"
        assert mtmd_mod._MTMD_DEVICE_ENV not in os.environ

    def test_env_is_cleared_even_when_the_native_call_raises(self, monkeypatch):
        """Process-global state, so the finally is load-bearing: a leaked value
        would silently pin every LATER mtmd load in this process too."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        lib = _FakeLib(fail=True)
        with pytest.raises(RuntimeError):
            _ctx(lib, 1)._open(use_gpu=True)
        assert lib.seen_env == "Vulkan1"          # it really was set before the raise
        assert mtmd_mod._MTMD_DEVICE_ENV not in os.environ

    def test_device_zero_never_sets_the_env(self, monkeypatch):
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        lib = _FakeLib()
        _ctx(lib, 0)._open(use_gpu=True)
        assert lib.seen_env is None
        assert mtmd_mod._MTMD_DEVICE_ENV not in os.environ

    def test_ambiguous_registry_never_sets_the_env(self, monkeypatch):
        """The guard, observed through _open rather than through the resolver -
        so a future refactor that resolves correctly but forgets to honour None
        still fails."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", IGPU)])
        lib = _FakeLib()
        _ctx(lib, 1)._open(use_gpu=True)
        assert lib.seen_env is None

    def test_cpu_attempt_never_sets_the_env(self, monkeypatch):
        """clip gates the env read on use_gpu, so the CPU attempt and
        retry_on_cpu() must not touch process-global state to no purpose."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        lib = _FakeLib()
        _ctx(lib, 1)._open(use_gpu=False)
        assert lib.seen_env is None
        assert mtmd_mod._MTMD_DEVICE_ENV not in os.environ

    def test_a_user_set_value_is_neither_overwritten_nor_deleted(self, monkeypatch):
        """never-override-user-selection: an explicit export outranks anything
        localm derives from config, and must still be there afterwards."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        monkeypatch.setenv(mtmd_mod._MTMD_DEVICE_ENV, "Vulkan0")
        lib = _FakeLib()
        _ctx(lib, 1)._open(use_gpu=True)
        assert lib.seen_env == "Vulkan0"                       # not overridden
        assert os.environ[mtmd_mod._MTMD_DEVICE_ENV] == "Vulkan0"   # not deleted

    def test_an_exported_empty_value_is_still_the_users(self, monkeypatch):
        """An exported-but-empty MTMD_BACKEND_DEVICE is the user's variable, and it
        is NOT equivalent to unset for clip (it takes the getenv branch, fails to
        init by that name, warns, and falls back). Deferring on PRESENCE rather
        than truthiness is also what makes the cleanup provably safe: localm only
        ever sets the key when it was absent, so it can never delete this."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        monkeypatch.setenv(mtmd_mod._MTMD_DEVICE_ENV, "")
        lib = _FakeLib()
        _ctx(lib, 1)._open(use_gpu=True)
        assert lib.seen_env == ""                                   # not overridden
        assert mtmd_mod._MTMD_DEVICE_ENV in os.environ              # not deleted

    def test_localm_mtmd_cpu_escape_hatch_still_short_circuits(self, monkeypatch):
        """The pre-existing opt-out must keep skipping the GPU attempt entirely,
        env resolution included."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        monkeypatch.setenv("LOCALM_MTMD_CPU", "1")
        lib = _FakeLib()
        assert _ctx(lib, 1)._open(use_gpu=True) is None
        assert lib.calls == 0
        assert mtmd_mod._MTMD_DEVICE_ENV not in os.environ


class TestConstructorAppliesItInTime:
    """Every test above sets ``_gpu_index`` by hand, so none of them can catch the
    constructor setting it too LATE - after ``_open`` has already run. This drives
    the real ``__init__``."""

    def _fake_lib_class(self, lib):
        lib.mtmd_support_vision = lambda ctx: True
        lib.mtmd_default_marker = lambda: b"<__media__>"
        return lib

    def test_real_init_pins_before_the_gpu_open(self, monkeypatch):
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        lib = self._fake_lib_class(_FakeLib())
        monkeypatch.setattr(mtmd_mod, "_load_lib", lambda: lib)
        # Skip the ABI probe: it needs a live native context, and the layout is a
        # property of the LIBRARY that this test has no stake in.
        monkeypatch.setattr(mtmd_mod, "_input_text_class", mtmd_mod._MtmdInputTextV2)

        ctx = mtmd_mod.MtmdContext("/fake/mmproj.gguf", 0xBEEF, gpu_index=1)

        assert ctx.on_gpu is True
        assert lib.seen_env == "Vulkan1", (
            "the constructor must set _gpu_index BEFORE _open runs; "
            f"the native call saw {lib.seen_env!r}")
        assert mtmd_mod._MTMD_DEVICE_ENV not in os.environ

    def test_real_init_with_default_index_pins_nothing(self, monkeypatch):
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        lib = self._fake_lib_class(_FakeLib())
        monkeypatch.setattr(mtmd_mod, "_load_lib", lambda: lib)
        monkeypatch.setattr(mtmd_mod, "_input_text_class", mtmd_mod._MtmdInputTextV2)

        mtmd_mod.MtmdContext("/fake/mmproj.gguf", 0xBEEF)

        assert lib.seen_env is None

    def test_unusable_index_warns_and_keeps_vision(self, monkeypatch, caplog):
        """A bad index must not cost the user vision, and must not be silent
        either (rule 5). Unreachable from llama.py, which passes an int derived
        from mp.main_gpu, so this pins the contract rather than a live path."""
        _devices(monkeypatch, [("CPU", CPU), ("Vulkan0", GPU), ("Vulkan1", GPU)])
        lib = self._fake_lib_class(_FakeLib())
        monkeypatch.setattr(mtmd_mod, "_load_lib", lambda: lib)
        monkeypatch.setattr(mtmd_mod, "_input_text_class", mtmd_mod._MtmdInputTextV2)

        with caplog.at_level("WARNING", logger="localm"):
            ctx = mtmd_mod.MtmdContext(
                "/fake/mmproj.gguf", 0xBEEF, gpu_index="not-a-device")

        assert ctx.on_gpu is True                 # vision survived
        assert lib.seen_env is None               # fell back to clip's own choice
        assert any("unusable projector device index" in r.getMessage()
                   for r in caplog.records), "the bad index was swallowed silently"


class TestLlamaCppPassesTheResolvedDevice:

    def test_load_mmproj_forwards_the_resolved_main_gpu(self, monkeypatch):
        """The wiring. mp.main_gpu is read AFTER apply_gpu_split, which forces it
        inside the configured split set - so this is the index the projector must
        follow, and it has to actually arrive at MtmdContext."""
        from tests._bare_llama import make_bare_llama
        seen = {}

        class _Recorder:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                seen["gpu_index"] = gpu_index
                self.supports_vision = True

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _Recorder)
        inst = make_bare_llama(_model_ptr=0xBEEF, _main_gpu_index=2)
        inst._load_mmproj("/fake/mmproj.gguf", verbose=True)
        assert seen["gpu_index"] == 2
        assert inst._mtmd is not None

    def test_load_mmproj_defaults_to_zero_when_unset(self, monkeypatch):
        """_load_mmproj is unit-tested against instances that never ran __init__,
        so the attribute can be absent; 0 is the leave-clip-alone value."""
        from localm.inference.backends.llamacpp import llama as llama_mod
        seen = {}

        class _Recorder:
            def __init__(self, mmproj_path, model_ptr, gpu_index=0):
                seen["gpu_index"] = gpu_index
                self.supports_vision = True

        monkeypatch.setattr(mtmd_mod, "MtmdContext", _Recorder)
        inst = llama_mod.LlamaCpp.__new__(llama_mod.LlamaCpp)
        inst._model_ptr = 0xBEEF
        inst._load_mmproj("/fake/mmproj.gguf", verbose=True)
        assert seen["gpu_index"] == 0
