# SPDX-License-Identifier: AGPL-3.0-or-later
"""Auto n_gpu_layers partial-offload (GgufBackend._auto_gpu_layers /
_effective_gpu_layers) and the model_meta layer-count cache.

The "needs partial CPU offload" badge promises a fallback the default load path
must actually deliver: with default n_gpu_layers=99, _check_vram charging the
FULL weight for any non-zero value refuses a too-big model. These tests pin the
mechanism that makes the promise true: when n_gpu_layers is left at 99 and auto
is on, the loader sizes how many layers fit from free VRAM so the model LOADS
(some layers on CPU) instead of raising, while an explicit -g is always honoured
verbatim.
"""

import json

import pytest
from unittest.mock import patch

from localm.inference.backends.gguf import GgufBackend


GB = 1024 ** 3


def _model(tmp_path, size_bytes, *, n_gpu_layers=99, auto=True, n_ctx=4096):
    # A tiny REAL file (so is_file() and the model_meta stat key work), with the
    # multi-GB on-disk size faked via _model_bytes. NEVER truncate() to GB sizes
    # here: Windows truncate() is not sparse and allocates real disk.
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\0" * 4096)
    b = GgufBackend(str(f), n_gpu_layers=n_gpu_layers,
                    n_gpu_layers_auto=auto, n_ctx=n_ctx)
    b._model_bytes = lambda: size_bytes
    return b


def _vram(free, total):
    """Patch VRAM measurement so a test gets a deterministic (free, total)
    reading regardless of the box this runs on.

    ``_free_vram_bytes`` prefers ``_free_total_vram_bytes`` (torch.cuda) but
    falls back to ``loader.gpu_memory_isolated()`` (the isolated VRAM-probe
    daemon) when torch cannot answer - on a box with a real GPU and a
    provisioned native runtime, that fallback can succeed for real and return
    genuine driver numbers, defeating a test that wants "VRAM is totally
    unmeasurable" (free=None). Both paths must be patched together, or
    `_vram(None, None)` is not actually unmeasurable on a GPU-equipped machine.

    A THIRD path has the same trap: ``_free_vram_bytes`` applies a device-global
    correction (``_device_global_free_bytes``) on a Windows + ROCm/HIP box,
    reading the box's REAL adapter usage via ADL/PDH and replacing the faked
    ``free`` with ``total - real_used``, which defeats both fakes above on
    exactly that hardware. Patched to None so the correction reports "not
    applicable" and the faked reading stands deterministically on every box."""
    from contextlib import ExitStack
    from localm.inference.backends.llamacpp import _loader

    stack = ExitStack()
    stack.enter_context(patch.object(
        GgufBackend, "_free_total_vram_bytes", return_value=(free, total)))
    stack.enter_context(patch.object(
        _loader, "gpu_memory_isolated",
        return_value=(None if free is None else (free, total))))
    stack.enter_context(patch.object(
        GgufBackend, "_device_global_free_bytes", return_value=None))
    return stack


# --------------------------------------------------------------------------- #
#  _auto_gpu_layers                                                            #
# --------------------------------------------------------------------------- #

class TestAutoGpuLayers:
    def test_full_offload_when_it_all_fits(self, tmp_path):
        b = _model(tmp_path, 8 * GB)
        with _vram(24 * GB, 24 * GB):
            assert b._auto_gpu_layers() == 99   # whole model + KV + overhead fit

    def test_partial_fraction_when_it_does_not_fit(self, tmp_path):
        # 8GB model, only 6GB free: some layers must move to CPU. With the assumed
        # 32-layer count (nothing cached), the count is floor(fraction * 32) and
        # lands strictly between 0 and 99.
        b = _model(tmp_path, 8 * GB)
        with _vram(6 * GB, 16 * GB):
            n = b._auto_gpu_layers()
        assert 0 < n < 99
        # The chosen offload's weight share must fit the GPU budget it was sized
        # against (free - KV - overhead), i.e. auto never over-commits VRAM.
        weight_budget = 6 * GB - 4096 * GgufBackend._bytes_per_token(8 * GB) \
            - GgufBackend._VRAM_OVERHEAD_BYTES
        assert (n / 32) * (8 * GB) <= weight_budget + 1  # +1 for int rounding

    def test_none_when_vram_unmeasurable(self, tmp_path):
        # Genuinely unmeasurable: NEITHER torch.cuda NOR the isolated native
        # probe can answer (e.g. no GPU present, or the probe daemon itself is
        # unreachable) -> (None, None). Auto must fall back honestly (return
        # None), never fabricate a precise offload.
        b = _model(tmp_path, 8 * GB)
        with _vram(None, None):
            assert b._auto_gpu_layers() is None

    def test_zero_at_the_extreme(self, tmp_path):
        # Free VRAM below even KV + overhead: nothing left for weights -> full CPU
        # (0), which still loads (slowly), the extreme end of the RAM offload.
        b = _model(tmp_path, 8 * GB)
        with _vram(1 * GB, 16 * GB):
            assert b._auto_gpu_layers() == 0

    def test_uses_cached_layer_count_when_available(self, tmp_path):
        # With a cached TRUE count of 80, the same 6GB/8GB fit scales by 80, not
        # the assumed 32 - a bigger, more precise offload count.
        b = _model(tmp_path, 8 * GB)
        with _vram(6 * GB, 16 * GB), \
             patch.object(GgufBackend, "_cached_layer_count", return_value=80):
            n = b._auto_gpu_layers()
        assert 0 < n < 99
        assert n > 32 * 0.4   # scaled by 80, materially larger than the /32 count


# --------------------------------------------------------------------------- #
#  _effective_gpu_layers                                                       #
# --------------------------------------------------------------------------- #

class TestEffectiveGpuLayers:
    def test_defers_to_explicit_gpu_layers(self, tmp_path, capsys):
        # A user who set -g 24 gets 24, even with auto on and a partial fit.
        b = _model(tmp_path, 8 * GB, n_gpu_layers=24, auto=True)
        with _vram(6 * GB, 16 * GB):
            assert b._effective_gpu_layers() == 24

    def test_auto_off_returns_configured(self, tmp_path):
        b = _model(tmp_path, 8 * GB, n_gpu_layers=99, auto=False)
        with _vram(6 * GB, 16 * GB):
            assert b._effective_gpu_layers() == 99

    def test_auto_on_default_sizes_partial_and_notifies(self, tmp_path, capsys):
        b = _model(tmp_path, 8 * GB, n_gpu_layers=99, auto=True)
        with _vram(6 * GB, 16 * GB):
            n = b._effective_gpu_layers()
        assert 0 < n < 99
        out = capsys.readouterr().out.lower()
        assert "gpu layers auto" in out          # mandatory notice printed
        assert "cpu" in out and "slower" in out

    def test_auto_full_fit_no_scary_notice(self, tmp_path, capsys):
        b = _model(tmp_path, 8 * GB, n_gpu_layers=99, auto=True)
        with _vram(24 * GB, 24 * GB):
            assert b._effective_gpu_layers() == 99
        assert "gpu layers auto" not in capsys.readouterr().out.lower()

    def test_unmeasurable_falls_back_quietly(self, tmp_path, capsys):
        # When neither VRAM path can answer (see _vram(None, None) above), full
        # offload via the display driver is the working default, and the
        # fallback prints no per-load console notice (it goes to debug).
        b = _model(tmp_path, 8 * GB, n_gpu_layers=99, auto=True)
        with _vram(None, None):
            assert b._effective_gpu_layers() == 99   # attempts the configured value
        out = capsys.readouterr().out.lower()
        assert "could not measure" not in out
        assert "gpu layers auto" not in out          # no scary partial-offload line


# --------------------------------------------------------------------------- #
#  _check_vram partial-awareness (the 0<g<99 path)                             #
# --------------------------------------------------------------------------- #

class TestCheckVramPartial:
    def test_auto_partial_that_fits_does_not_raise(self, tmp_path, capsys):
        # An auto-sized partial load PASSES the preflight: only the offloaded
        # fraction is charged.
        b = _model(tmp_path, 12 * GB, n_gpu_layers=99, auto=True)
        with _vram(8 * GB, 16 * GB):
            b.effective_gpu_layers = b._effective_gpu_layers()   # resolves partial
            assert 0 < b.effective_gpu_layers < 99
            b._check_vram()                                       # must not raise
        assert "Low VRAM" not in capsys.readouterr().out

    def test_explicit_partial_charges_only_fraction(self, tmp_path):
        # -g 16 on a 20GB model: charging the FULL weight would exceed 16GB
        # total and raise; charging half (16/32) fits and must not.
        b = _model(tmp_path, 20 * GB, n_gpu_layers=16, auto=False)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15 * GB), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16 * GB):
            b._check_vram()   # must not raise (half of 20GB fits)

    def test_explicit_partial_still_raises_when_truly_too_big(self, tmp_path):
        # Even half of a 20GB model (10GB) + KV + 1.5GB overhead > an 8GB card:
        # a partial that genuinely cannot fit must still be refused.
        b = _model(tmp_path, 20 * GB, n_gpu_layers=16, auto=False)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=7 * GB), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=8 * GB):
            with pytest.raises(RuntimeError, match="cannot fit regardless"):
                b._check_vram()

    def test_pinned_full_that_cannot_fit_still_raises(self, tmp_path):
        # auto OFF, full offload (99), model bigger than the card: refused.
        b = _model(tmp_path, 20 * GB, n_gpu_layers=99, auto=False)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15 * GB), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16 * GB):
            with pytest.raises(RuntimeError, match="cannot fit regardless"):
                b._check_vram()

    def test_refusal_message_mentions_auto_option(self, tmp_path):
        b = _model(tmp_path, 20 * GB, n_gpu_layers=99, auto=False)
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=15 * GB), \
             patch.object(GgufBackend, "_total_vram_bytes", return_value=16 * GB):
            with pytest.raises(RuntimeError) as exc:
                b._check_vram()
        assert "n_gpu_layers_auto" in str(exc.value)


# --------------------------------------------------------------------------- #
#  _check_context_fit must honour the RESOLVED offload count (the grow twin)   #
# --------------------------------------------------------------------------- #

class TestCheckContextFitAutoAware:
    def test_auto_resolved_cpu_load_does_not_act_on_grow(self, tmp_path):
        # An auto-sized CPU-only load (effective 0) still carries
        # n_gpu_layers==99, so the grow check gates on effective_gpu_layers and
        # early-returns None, leaving the target as-is.
        b = _model(tmp_path, 12 * GB, n_gpu_layers=99, auto=True)
        b.effective_gpu_layers = 0                     # what load() resolves to
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=100 * 1024 ** 2):
            assert b._check_context_fit(8192, current_ctx=4096) is None   # CPU-only

    def test_partial_offload_charges_only_kv_fraction_on_grow(self, tmp_path):
        # A partial offload keeps only its layers' KV in VRAM, so only that
        # fraction of the KV delta is charged.
        b = _model(tmp_path, 20 * GB, n_gpu_layers=99, auto=True)
        b.effective_gpu_layers = 16                    # 16/32 assumed -> half KV
        bpt = GgufBackend._bytes_per_token(20 * GB)
        delta_full = (8192 - 4096) * bpt               # if it charged the FULL fraction
        free = int(delta_full * 0.5 + 10 * 1024 ** 2)  # room for the half-delta, not the full
        assert delta_full > free                        # full-fraction delta would not fit...
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=free):
            # ...but half the delta fits VRAM, so keep the KV cache there (True).
            assert b._check_context_fit(8192, current_ctx=4096) is True

    def test_full_offload_uses_ram_for_an_oversize_grow_instead_of_raising(self, tmp_path):
        # A full/pinned load (effective 99) charges the whole KV delta; an oversize
        # grow that will not fit VRAM keeps the full window with its KV cache in
        # system RAM (return False) - a degrade, NOT a raise, NOT a shrink.
        b = _model(tmp_path, 20 * GB, n_gpu_layers=99, auto=False)
        b.effective_gpu_layers = 99
        bpt = GgufBackend._bytes_per_token(20 * GB)
        free = int((8192 - 4096) * bpt * 0.5)          # not enough for the full delta
        with patch.object(GgufBackend, "_free_vram_bytes", return_value=free):
            assert b._check_context_fit(8192, current_ctx=4096) is False


# --------------------------------------------------------------------------- #
#  load() end-to-end: resolve once, partial load is not refused                #
# --------------------------------------------------------------------------- #

class TestLoadResolvesGpuLayers:
    def test_load_sizes_partial_and_does_not_refuse(self, tmp_path):
        # A too-big model with auto on default must reach _load_native (partial),
        # not raise in _check_vram. _load_native is stubbed so no native runtime
        # is needed.
        b = _model(tmp_path, 12 * GB, n_gpu_layers=99, auto=True)
        with _vram(8 * GB, 16 * GB), \
             patch.object(GgufBackend, "_load_native", lambda self: None):
            b.load()
        assert 0 < b.effective_gpu_layers < 99   # resolved once, shared with load


# --------------------------------------------------------------------------- #
#  Post-load layer-count cache write                                           #
# --------------------------------------------------------------------------- #

class TestPostLoadLayerCountCache:
    def test_load_native_caches_true_layer_count(self, tmp_path, monkeypatch):
        """The real load happens in the isolated worker (see
        llamacpp/_runner.py): it reports n_layers back in the load response and
        GgufBackend._load_native persists it, with the disk write staying
        parent-side."""
        import localm.config as cfg
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)

        b = _model(tmp_path, 1_000_000, n_gpu_layers=99, auto=False)
        with patch("localm.discover.list_gpus", return_value=([], "ok")), \
             patch("localm.inference.backends.llamacpp._runner.ModelRunner.spawn_and_load",
                   return_value={"n_layers": 42, "kv_bytes_per_token": 0,
                                 "supports_images": False}):
            b._load_native()

        from localm.model_meta import cached_n_layers
        assert cached_n_layers(b.model_path) == 42
        # Persisted under the data home, keyed to this model.
        assert (tmp_path / "model_meta.json").is_file()


# --------------------------------------------------------------------------- #
#  model_meta cache unit behaviour                                             #
# --------------------------------------------------------------------------- #

class TestModelMetaCache:
    def _f(self, tmp_path, size=1000):
        f = tmp_path / "m.gguf"
        with open(f, "wb") as fh:
            fh.truncate(size)
        return str(f)

    def test_store_and_read_roundtrip(self, tmp_path, monkeypatch):
        import localm.config as cfg
        import localm.model_meta as meta
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        path = self._f(tmp_path)
        assert meta.cached_n_layers(path) is None        # nothing cached yet
        meta.store_n_layers(path, 32)
        assert meta.cached_n_layers(path) == 32

    def test_cache_misses_when_file_changes(self, tmp_path, monkeypatch):
        import localm.config as cfg
        import localm.model_meta as meta
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        path = self._f(tmp_path, size=1000)
        meta.store_n_layers(path, 32)
        # A different size (a replaced/rebuilt model) invalidates the entry.
        with open(path, "wb") as fh:
            fh.truncate(2000)
        assert meta.cached_n_layers(path) is None

    def test_corrupt_cache_returns_none_not_crash(self, tmp_path, monkeypatch):
        import localm.config as cfg
        import localm.model_meta as meta
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        (tmp_path / "model_meta.json").write_text("{not json", encoding="utf-8")
        path = self._f(tmp_path)
        assert meta.cached_n_layers(path) is None         # tolerated, logged
        # And a later store still works (overwrites the corrupt file atomically).
        meta.store_n_layers(path, 16)
        assert meta.cached_n_layers(path) == 16

    def test_ignores_non_positive_counts(self, tmp_path, monkeypatch):
        import localm.config as cfg
        import localm.model_meta as meta
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        path = self._f(tmp_path)
        meta.store_n_layers(path, 0)
        meta.store_n_layers(path, -5)
        assert meta.cached_n_layers(path) is None
        assert not (tmp_path / "model_meta.json").exists()

    def test_entries_capped(self, tmp_path, monkeypatch):
        import localm.config as cfg
        import localm.model_meta as meta
        monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
        monkeypatch.setattr(meta, "_MAX_ENTRIES", 4)
        # Store more distinct models than the cap allows; the file must not grow
        # past the cap (oldest dropped).
        for i in range(10):
            f = tmp_path / f"m{i}.gguf"
            with open(f, "wb") as fh:
                fh.truncate(1000 + i)
            meta.store_n_layers(str(f), 10 + i)
        data = json.loads((tmp_path / "model_meta.json").read_text(encoding="utf-8"))
        assert len(data) == 4


class TestVramOverheadConfigResolution:
    """vram_overhead_mb (config.py) -> GgufBackend._VRAM_OVERHEAD_BYTES, wired
    through localm.inference.engine.create_backend()'s
    _resolve_vram_overhead_bytes helper. Normal writes (PATCH /v1/config,
    `localm config`) already enforce a valid int via
    settings_schema.validate_update, but a hand-edited config.json is not
    type-checked on load (config.py's load_config just merges the stored
    dict) - a present-but-unparseable value must fall back to the built-in
    default instead of crashing create_backend() (and therefore every
    subsequent model load) with an uncaught ValueError/TypeError."""

    def test_valid_value_overrides_the_default(self):
        from localm.inference.engine import _resolve_vram_overhead_bytes
        assert _resolve_vram_overhead_bytes({"vram_overhead_mb": 900}) == 900 * 1024 ** 2

    def test_missing_key_uses_the_built_in_default(self):
        from localm.inference.engine import _resolve_vram_overhead_bytes
        from localm.vram import VRAM_OVERHEAD_BYTES
        assert _resolve_vram_overhead_bytes({}) == VRAM_OVERHEAD_BYTES

    @pytest.mark.parametrize("bad_value", ["bogus", [], {}, object()])
    def test_unparseable_value_falls_back_instead_of_raising(self, bad_value):
        from localm.inference.engine import _resolve_vram_overhead_bytes
        from localm.vram import VRAM_OVERHEAD_BYTES
        assert _resolve_vram_overhead_bytes(
            {"vram_overhead_mb": bad_value}) == VRAM_OVERHEAD_BYTES

    def test_create_backend_does_not_crash_on_a_hand_edited_bad_value(
            self, tmp_path, monkeypatch):
        """End-to-end: a malformed value that only a hand-edited config.json
        (never the validated PATCH/CLI paths) could produce must not brick
        create_backend(). A ValueError/TypeError propagates uncaught from
        engine.py before switch_engine's own RuntimeError->503 handling runs,
        since it is not a RuntimeError and fires earlier."""
        from localm.inference.engine import create_backend
        model = tmp_path / "model.gguf"
        model.write_bytes(b"x")
        from localm.config import load_config as real_load_config
        base = real_load_config()
        monkeypatch.setattr(
            "localm.inference.engine.load_config",
            lambda: {**base, "vram_overhead_mb": "bogus"})
        backend = create_backend(str(model))
        from localm.vram import VRAM_OVERHEAD_BYTES
        assert backend._VRAM_OVERHEAD_BYTES == VRAM_OVERHEAD_BYTES


# --------------------------------------------------------------------------- #
#  ctx_auto embedder-footprint reservation                                     #
# --------------------------------------------------------------------------- #

class TestAutoCtxEmbedderReservation:
    """_auto_ctx_max must hold back room for the CONFIGURED embedder so the
    context window sized at chat-load time does not claim the VRAM the embedder
    needs when it loads later (first memory/RAG use). Oversubscribing collapses
    generation on a real 16 GB card, because WDDM pages the overcommit to system
    RAM.

    _auto_gpu_layers is NOT reservation-aware: chat weights keep VRAM priority
    (a partial chat offload to protect the embedder would slow the primary
    workload); the context window is the flexible resource."""

    def test_reservation_shrinks_the_auto_ctx_ceiling(self, tmp_path, monkeypatch):
        from localm.inference.backends.llamacpp import _sizing
        b = _model(tmp_path, 8 * GB)
        overhead = b._VRAM_OVERHEAD_BYTES
        bpt = b._bytes_per_token(8 * GB)

        monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes", lambda: 0)
        with _vram(12 * GB, 16 * GB):
            base = b._auto_ctx_max()
        expected_base = ((12 * GB - 8 * GB - overhead) // bpt) // 1024 * 1024
        assert base == expected_base

        reservation = int(1.2 * GB)
        monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes",
                            lambda: reservation)
        with _vram(12 * GB, 16 * GB):
            reserved = b._auto_ctx_max()
        expected = ((12 * GB - 8 * GB - overhead - reservation) // bpt) // 1024 * 1024
        assert reserved == expected
        assert reserved < base

    def test_zero_reservation_leaves_ceiling_unchanged(self, tmp_path, monkeypatch):
        from localm.inference.backends.llamacpp import _sizing
        b = _model(tmp_path, 8 * GB)
        monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes", lambda: 0)
        with _vram(12 * GB, 16 * GB):
            first = b._auto_ctx_max()
        with _vram(12 * GB, 16 * GB):
            second = b._auto_ctx_max()
        assert first == second

    def test_gpu_layers_ignore_the_reservation(self, tmp_path, monkeypatch):
        """Weights keep priority: a huge reservation must not shrink the
        offloaded layer count."""
        from localm.inference.backends.llamacpp import _sizing
        b = _model(tmp_path, 8 * GB)
        with _vram(12 * GB, 16 * GB):
            monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes",
                                lambda: 0)
            base_layers = b._auto_gpu_layers()
            monkeypatch.setattr(_sizing, "embedder_ctx_reservation_bytes",
                                lambda: 8 * GB)
            assert b._auto_gpu_layers() == base_layers


class TestEmbedderCtxReservationBytes:
    """The reservation source itself: configured-but-unloaded embedder's file
    size + the codebase's standing 20% slop; 0 in every case where reserving
    would be wrong, and 0 (never a crash) on any failure."""

    def test_zero_when_embedder_already_loaded(self, monkeypatch, tmp_path):
        from localm.inference.backends.llamacpp import _sizing
        f = tmp_path / "emb.gguf"
        f.write_bytes(b"\0" * 1000)
        monkeypatch.setattr("localm.inference.embedder.loaded_path",
                            lambda: str(f))
        assert _sizing.embedder_ctx_reservation_bytes() == 0

    def test_zero_when_no_model_resolvable(self, monkeypatch):
        monkeypatch.setattr("localm.inference.embedder.loaded_path", lambda: None)
        monkeypatch.setattr(
            "localm.inference.embedder.resolve_embedding_model_path",
            lambda **kw: None)
        from localm.inference.backends.llamacpp import _sizing
        assert _sizing.embedder_ctx_reservation_bytes() == 0

    def test_reserves_file_size_plus_slop(self, monkeypatch, tmp_path):
        f = tmp_path / "emb.gguf"
        f.write_bytes(b"\0" * 1000)
        monkeypatch.setattr("localm.inference.embedder.loaded_path", lambda: None)
        monkeypatch.setattr(
            "localm.inference.embedder.resolve_embedding_model_path",
            lambda **kw: str(f))
        from localm.inference.backends.llamacpp import _sizing
        assert _sizing.embedder_ctx_reservation_bytes() == 1200

    def test_zero_on_resolver_failure(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("resolver exploded")
        monkeypatch.setattr("localm.inference.embedder.loaded_path", lambda: None)
        monkeypatch.setattr(
            "localm.inference.embedder.resolve_embedding_model_path", _boom)
        from localm.inference.backends.llamacpp import _sizing
        assert _sizing.embedder_ctx_reservation_bytes() == 0
