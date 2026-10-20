# SPDX-License-Identifier: AGPL-3.0-or-later
"""The kv_cache_type setting: f16 (default) or q8_0. Every llama context a chat
model creates carries the type, the KV sizing charges 1.0625 bytes per element
for q8_0, and a context llama.cpp refuses under q8_0 names the setting."""

import ast
import ctypes
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from localm import kv_cache
from localm import settings_schema as ss
from localm.config import DEFAULT_CONFIG
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp import _draftmodel, _structs
from localm.inference.backends.llamacpp import llama as llama_mod
from localm.inference.backends.llamacpp.llama import LlamaCpp
from localm.model_manager.gguf import (
    gguf_kv_bytes_per_token, gguf_mtp_draft_kv_bytes_per_token)
from localm.sysstats import estimate_vram
from tests._bare_llama import make_bare_llama
from tests.test_auto_gpu_layers import _vram
from tests.test_kv_bytes_from_gguf import _gguf, _shape

GB = 1024 ** 3
GGML_F16 = 1
GGML_Q8_0 = 8
API = "localm.inference.backends.llamacpp.llama.api"


# --------------------------------------------------------------------------- #
#  The element sizes                                                          #
# --------------------------------------------------------------------------- #

class TestBytes:
    def test_q8_0_charges_34_bytes_per_32_elements(self):
        # 131072 f16 bytes = 65536 elements; q8_0 = 65536 / 32 * 34.
        assert kv_cache.kv_bytes_for_type(131072, "q8_0") == 65536 // 32 * 34 == 69632

    def test_q8_0_is_1_0625_bytes_per_element(self):
        f16 = 57344   # a 7B dense model: 28 layers * 4 kv heads * 128 * 2 * 2
        assert kv_cache.kv_bytes_for_type(f16, "q8_0") == f16 // 2 * 1.0625

    def test_f16_and_none_are_unchanged(self):
        assert kv_cache.kv_bytes_for_type(57344, "f16") == 57344
        assert kv_cache.kv_bytes_for_type(57344, None) == 57344

    def test_a_partial_block_rounds_up(self):
        # 3 f16 bytes = 1.5 elements -> 1.59375 q8_0 bytes, charged as 2.
        assert kv_cache.kv_bytes_for_type(3, "q8_0") == 2

    def test_zero_stays_zero(self):
        assert kv_cache.kv_bytes_for_type(0, "q8_0") == 0

    @pytest.mark.parametrize("bad", ["q4_0", "Q8_0", "", "f32"])
    def test_an_unknown_type_raises(self, bad):
        with pytest.raises(ValueError, match="kv_cache_type"):
            kv_cache.kv_bytes_for_type(1000, bad)

    def test_ggml_ids(self):
        assert kv_cache.ggml_type_id("f16") == GGML_F16
        assert kv_cache.ggml_type_id("q8_0") == GGML_Q8_0
        with pytest.raises(ValueError, match="kv_cache_type"):
            kv_cache.ggml_type_id("q4_0")

    @pytest.mark.integration
    def test_ggml_type_ids_match_the_runtime(self):
        """The ids and block sizes are the shipped runtime's own answer."""
        from tests._real_gguf import native_runtime_lib_path, require_native_runtime
        lib_path = native_runtime_lib_path()
        if lib_path is None:
            pytest.skip("native llama runtime not provisioned (run 'localm setup-llama')")
        require_native_runtime()
        base = next(Path(lib_path).parent.glob("ggml-base*"), None)
        if base is None:
            pytest.skip("no ggml-base library next to the runtime")
        lib = ctypes.CDLL(str(base))
        lib.ggml_type_name.restype = ctypes.c_char_p
        lib.ggml_type_size.restype = ctypes.c_size_t
        lib.ggml_blck_size.restype = ctypes.c_int64
        for name in kv_cache.KV_CACHE_TYPES:
            tid = kv_cache.ggml_type_id(name)
            assert lib.ggml_type_name(tid).decode() == name
            per_block = 1024 * lib.ggml_blck_size(tid)
            assert kv_cache.kv_bytes_for_type(2 * per_block, name) == (
                per_block // lib.ggml_blck_size(tid) * lib.ggml_type_size(tid))


class TestResolve:
    def test_absent_is_f16(self):
        assert kv_cache.resolve_kv_cache_type({}) == "f16"

    @pytest.mark.parametrize("kind", ["f16", "q8_0"])
    def test_a_known_type_is_used(self, kind):
        assert kv_cache.resolve_kv_cache_type({"kv_cache_type": kind}) == kind

    @pytest.mark.parametrize("bad", ["q4_0", "Q8_0", 8, None, ""])
    def test_an_unknown_stored_value_warns_and_uses_f16(self, bad):
        with patch("localm.debuglog.logger.warning") as warn:
            assert kv_cache.resolve_kv_cache_type({"kv_cache_type": bad}) == "f16"
        warn.assert_called_once()
        assert "kv_cache_type" in warn.call_args.args[0]


# --------------------------------------------------------------------------- #
#  Config, schema, fingerprint                                                #
# --------------------------------------------------------------------------- #

class TestSetting:
    def test_the_default_is_f16(self):
        assert DEFAULT_CONFIG["kv_cache_type"] == "f16"

    def test_the_schema_field_offers_exactly_the_types(self):
        field = next(f for f in ss.CORE_FIELDS if f.key == "kv_cache_type")
        assert field.widget == ss.Widget.SELECT
        assert field.options == list(kv_cache.KV_CACHE_TYPES) == ["f16", "q8_0"]
        assert field.applies == ss.Applies.NEXT_LOAD
        assert field.group == "Engine"

    @pytest.mark.parametrize("kind", ["f16", "q8_0"])
    def test_every_type_validates(self, kind):
        assert ss.validate_update({"kv_cache_type": kind}) == {"kv_cache_type": kind}

    @pytest.mark.parametrize("bad", ["q4_0", "Q8_0", "", True, 8])
    def test_an_unknown_type_is_rejected(self, bad):
        with pytest.raises(ValueError, match="kv_cache_type"):
            ss.validate_update({"kv_cache_type": bad})

    def test_it_is_in_the_load_fingerprint_keys(self):
        from localm.inference.routing_latch import LOAD_CONFIG_KEYS
        assert "kv_cache_type" in LOAD_CONFIG_KEYS

    def test_it_is_in_the_bug_report_config_keys(self):
        from localm.bugreport import _SAFE_CONFIG_KEYS
        assert "kv_cache_type" in _SAFE_CONFIG_KEYS


# --------------------------------------------------------------------------- #
#  Plumbing: config -> backend -> worker -> LlamaCpp                          #
# --------------------------------------------------------------------------- #

class TestPlumbing:
    def test_create_backend_reads_the_setting(self, tmp_path):
        from localm.inference import engine
        f = tmp_path / "m.gguf"
        f.write_bytes(b"GGUF" + b"\0" * 64)
        cfg = dict(DEFAULT_CONFIG, kv_cache_type="q8_0")
        with patch.object(engine, "load_config", return_value=cfg):
            backend = engine.create_backend(str(f))
        assert backend.kv_cache_type == "q8_0"

    def test_the_backend_refuses_an_unknown_type(self, tmp_path):
        with pytest.raises(ValueError, match="kv_cache_type"):
            GgufBackend(str(tmp_path / "m.gguf"), kv_cache_type="q4_0")

    @pytest.mark.parametrize("kind", ["q8_0", None])
    def test_the_worker_hands_the_type_to_llamacpp(self, tmp_path, kind):
        import sys
        from tests.test_gguf_worker import _StubLlm, _worker
        w = _worker(str(tmp_path / "m.gguf"), kv_cache_type=kind)
        fake = MagicMock()
        fake.LlamaCpp.return_value = _StubLlm()
        with patch("localm.inference.backends.llamacpp._loader.load_lib"),              patch.dict(sys.modules, {"localm.inference.backends.llamacpp": fake}):
            w.load()
        _, kwargs = fake.LlamaCpp.call_args
        assert kwargs.get("kv_cache_type") == kind

    @pytest.mark.parametrize("kind,sent", [("q8_0", "q8_0"), ("f16", None)])
    def test_the_backend_sends_the_type_to_the_worker(self, tmp_path, kind, sent):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        b = GgufBackend(str(f), kv_cache_type=kind)
        with patch("localm.config.load_config", return_value=dict(DEFAULT_CONFIG)),              patch("localm.discover.list_gpus", return_value=([], "ok")),              patch("localm.inference.backends.llamacpp._runner.ModelRunner."
                   "spawn_and_load", return_value={}) as spawn:
            b.effective_gpu_layers = 0
            b._load_native()
        assert spawn.call_args[0][0].get("kv_cache_type") == sent

    def test_llamacpp_refuses_an_unknown_type_before_any_native_call(self):
        api = MagicMock()
        with patch(API, api), pytest.raises(ValueError, match="kv_cache_type"):
            LlamaCpp("m.gguf", n_ctx=512, kv_cache_type="q4_0")
        api.llama_load_model_from_file.assert_not_called()


# --------------------------------------------------------------------------- #
#  Every context creation carries the type                                    #
# --------------------------------------------------------------------------- #

def _context_params():
    """A real native context-params struct (the V3 layout) with llama.cpp's
    own defaults for the two fields under test."""
    cp = _structs.LlamaContextParamsV3()
    cp.type_k = GGML_F16
    cp.type_v = GGML_F16
    return cp


def _recording_api(init_results=None):
    """A fake api whose llama_init_from_model records the params it was given
    and returns *init_results* in turn (a truthy pointer by default)."""
    api = MagicMock()
    seen = []
    api.llama_context_default_params.side_effect = _context_params
    results = iter(init_results or [])

    def init(model, cp):
        seen.append((cp.type_k, cp.type_v))
        return next(results, ctypes.c_void_p(2))

    api.llama_init_from_model.side_effect = init
    api.llama_decode.return_value = 0
    api.llama_memory_seq_rm.return_value = True
    return api, seen


def _load_api(init_results=None):
    api, seen = _recording_api(init_results)
    api.llama_model_default_params.return_value = SimpleNamespace(
        main_gpu=0, n_gpu_layers=0, use_mmap=True, load_mtp=False)
    api.has_lora_api.return_value = False
    api.has_encoder_api.return_value = True
    api.llama_model_has_encoder.return_value = False
    api.llama_model_has_decoder.return_value = True
    api.llama_load_model_from_file.return_value = ctypes.c_void_p(1)
    return api, seen


class TestEveryContextCarriesTheType:
    @pytest.mark.parametrize("kind,want", [("q8_0", GGML_Q8_0), ("f16", GGML_F16),
                                           (None, GGML_F16)])
    def test_the_initial_context(self, kind, want):
        api, seen = _load_api()
        with patch(API, api), \
             patch.object(llama_mod.pretokenizer_guard, "read_pre_type", return_value=None):
            LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=0, verbose=True,
                     kv_cache_type=kind).close()
        assert seen and seen[0] == (want, want)

    @pytest.mark.parametrize("kind,want", [("q8_0", GGML_Q8_0), (None, GGML_F16)])
    def test_the_regrown_context(self, kind, want):
        llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, _ctx_capacity=512,
                              _kv_cache_type=kind, _offload_kqv=True)
        api, seen = _recording_api()
        with patch(API, api):
            try:
                llm._prefill_fresh_context([1, 2, 3], needed=4096)
            except Exception:
                pass
        assert seen and seen[0] == (want, want)

    @pytest.mark.parametrize("kind,want", [("q8_0", GGML_Q8_0), (None, GGML_F16)])
    def test_the_slots_context(self, kind, want):
        llm = make_bare_llama(_model_ptr=111, _ctx_ptr=None, n_parallel=2,
                              _kv_cache_type=kind)
        api, seen = _recording_api()
        with patch(API, api):
            llm._recreate_slots_context(4096, True)
        assert seen == [(want, want)]

    @pytest.mark.parametrize("kind,want", [("q8_0", GGML_Q8_0), (None, GGML_F16)])
    def test_the_mtp_draft_context(self, kind, want):
        llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, _kv_cache_type=kind)
        api, seen = _recording_api()
        api.llama_model_mtp_support.return_value = (True, "ok")
        api.llama_set_embeddings_nextn.return_value = False
        with patch(API, api):
            llm._create_mtp_context(4096)
        assert seen == [(want, want)]

    @pytest.mark.parametrize("kind,want", [("q8_0", GGML_Q8_0), (None, GGML_F16)])
    def test_the_draft_model_context(self, kind, want):
        llm = make_bare_llama(_kv_cache_type=kind)
        source = _draftmodel.DraftModelSource(llm, ctypes.c_void_p(5), 4)
        api, seen = _recording_api()
        with patch(API, api), patch.object(llama_mod, "_greedy_chain", return_value=7):
            assert source.create_context(4096, True) == ""
        assert seen == [(want, want)]

    def test_no_chat_context_is_created_without_applying_the_type(self):
        """Every function in the chat backend that asks for context params
        also applies the KV cache type, so a new creation site cannot ship
        without it."""
        missing = []
        for module in (llama_mod, _draftmodel):
            tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                calls = {getattr(n.func, "attr", getattr(n.func, "id", None))
                         for n in ast.walk(fn) if isinstance(n, ast.Call)}
                if ("llama_context_default_params" in calls
                        and "apply_kv_cache_type" not in calls):
                    missing.append(f"{module.__name__}.{fn.name}")
        assert not missing, f"context params created without the KV type: {missing}"


# --------------------------------------------------------------------------- #
#  A refused context names the setting                                        #
# --------------------------------------------------------------------------- #

class TestRefusalNamesTheSetting:
    def test_the_initial_context(self):
        api, _seen = _load_api(init_results=[None])
        with patch(API, api), \
             patch.object(llama_mod.pretokenizer_guard, "read_pre_type", return_value=None), \
             pytest.raises(RuntimeError) as err:
            LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=0, verbose=True,
                     kv_cache_type="q8_0")
        msg = str(err.value)
        assert msg.startswith("Failed to create llama context")
        assert "kv_cache_type" in msg and "q8_0" in msg and "f16" in msg

    def test_the_initial_context_under_f16_does_not_mention_it(self):
        api, _seen = _load_api(init_results=[None])
        with patch(API, api), \
             patch.object(llama_mod.pretokenizer_guard, "read_pre_type", return_value=None), \
             pytest.raises(RuntimeError) as err:
            LlamaCpp("m.gguf", n_ctx=512, n_gpu_layers=0, verbose=True)
        assert "kv_cache_type" not in str(err.value)

    def test_the_regrown_context(self):
        llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222, _ctx_capacity=512,
                              _kv_cache_type="q8_0")
        api, _seen = _recording_api(init_results=[None])
        with patch(API, api), pytest.raises(RuntimeError) as err:
            llm._prefill_fresh_context([1, 2, 3], needed=4096)
        assert "kv_cache_type" in str(err.value)

    def test_the_slots_context(self):
        llm = make_bare_llama(_model_ptr=111, _ctx_ptr=None, n_parallel=2,
                              _kv_cache_type="q8_0")
        api, _seen = _recording_api(init_results=[None])
        with patch(API, api), pytest.raises(RuntimeError) as err:
            llm._recreate_slots_context(4096, True)
        assert "kv_cache_type" in str(err.value)

    def test_the_load_failure_text_keeps_the_note(self):
        from localm.inference.backends.gguf import _load_failure_message
        reason = RuntimeError("Failed to create llama context."
                              + kv_cache.context_refused_note("q8_0"))
        msg = _load_failure_message(reason)
        assert "kv_cache_type" in msg
        assert "The weights loaded" in msg


# --------------------------------------------------------------------------- #
#  Sizing                                                                     #
# --------------------------------------------------------------------------- #

class TestSizing:
    def test_the_header_probe(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        f16 = 32 * 8 * 128 * 2 * 2
        assert gguf_kv_bytes_per_token(f) == f16
        assert gguf_kv_bytes_per_token(f, "f16") == f16
        assert gguf_kv_bytes_per_token(f, "q8_0") == 32 * 8 * 128 * 2 * 1.0625

    def test_the_mtp_draft_header_probe(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("qwen35", 48, 2048, 16, 4))
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1) == 2048
        assert gguf_mtp_draft_kv_bytes_per_token(f, 1, "q8_0") == 1088

    def test_the_loaded_model_figure(self):
        api = MagicMock()
        api.has_kv_head_api.return_value = True
        api.has_hybrid_api.return_value = False
        api.llama_model_n_embd.return_value = 4096
        api.llama_model_n_head.return_value = 32
        api.llama_model_n_head_kv.return_value = 8
        llm = make_bare_llama(_model_ptr=111, n_layers=32, is_encoder_decoder=False)
        with patch(API, api):
            f16 = llm._read_kv_bytes_per_token()
        assert f16 == 32 * 8 * 128 * 2 * 2
        assert kv_cache.kv_bytes_for_type(f16, "q8_0") == f16 * 17 // 32
        src = inspect.getsource(LlamaCpp.__init__)
        assert "kv_bytes_for_type(self._read_kv_bytes_per_token(), self._kv_cache_type)" in src

    @pytest.mark.parametrize("kind,want", [("f16", 131072), ("q8_0", 69632)])
    def test_the_backend_charges_the_type(self, tmp_path, kind, want):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        b = GgufBackend(str(f), kv_cache_type=kind)
        assert b._kv_bytes_per_token() == want

    def test_the_size_class_fallback_is_scaled_too(self, tmp_path):
        f = tmp_path / "m.gguf"
        f.write_bytes(b"\0" * 4096)
        f16 = GgufBackend(str(f))
        q8 = GgufBackend(str(f), kv_cache_type="q8_0")
        for b in (f16, q8):
            b._model_bytes = lambda: 8 * GB
        base = f16._kv_bytes_per_token()
        assert base == GgufBackend._bytes_per_token(8 * GB)
        assert q8._kv_bytes_per_token() == -(-base * 17 // 32)

    def test_the_auto_context_ceiling_grows_with_q8_0(self, tmp_path):
        f = _gguf(tmp_path / "m.gguf", _shape("llama", 32, 4096, 32, 8))
        ceilings = {}
        for kind in ("f16", "q8_0"):
            b = GgufBackend(str(f), ctx_auto=True, kv_cache_type=kind)
            b._model_bytes = lambda: 4 * GB
            with _vram(8 * GB, 16 * GB), \
                 patch("localm.inference.backends.llamacpp._sizing."
                       "embedder_ctx_reservation_bytes", return_value=0):
                ceilings[kind] = b._auto_ctx_max(capped=False)
        assert ceilings["q8_0"] > ceilings["f16"]
        assert ceilings["q8_0"] / ceilings["f16"] == pytest.approx(32 / 17, rel=0.02)

    def test_estimate_vram_scales_the_header_figure(self):
        f16 = estimate_vram(4 * GB, 8192, kv_bytes_per_token=131072)
        q8 = estimate_vram(4 * GB, 8192, kv_bytes_per_token=131072, kv_cache_type="q8_0")
        assert f16["kv_cache"] == 8192 * 131072
        assert q8["kv_cache"] == 8192 * 69632
        assert q8["weights"] == f16["weights"]

    def test_estimate_vram_scales_the_size_class_fallback(self):
        f16 = estimate_vram(8 * GB, 8192)
        q8 = estimate_vram(8 * GB, 8192, kv_cache_type="q8_0")
        bpt = GgufBackend._bytes_per_token(8 * GB)
        assert f16["kv_cache"] == 8192 * bpt
        assert q8["kv_cache"] == 8192 * -(-bpt * 17 // 32)
