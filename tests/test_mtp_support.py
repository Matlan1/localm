# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Multi-Token Prediction (MTP) model support."""

import ctypes
from unittest.mock import MagicMock, patch

from localm.config import DEFAULT_CONFIG
from localm.settings_schema import CORE_FIELDS
from localm.inference.backends.base import BaseBackend
from localm.inference.backends.gguf import GgufBackend
from localm.inference.backends.llamacpp._structs import (
    LLAMA_CONTEXT_TYPE_DEFAULT,
    LLAMA_CONTEXT_TYPE_MTP,
    LlamaModelParamsV2,
)
from localm.inference.backends.llamacpp import _api as api
from localm.inference.backends.llamacpp.llama import LlamaCpp
from localm.inference.engine import Engine


def test_mtp_constants_and_structs():
    """Verify MTP context type constants and model param struct offsets."""
    assert LLAMA_CONTEXT_TYPE_DEFAULT == 0
    assert LLAMA_CONTEXT_TYPE_MTP == 1
    assert hasattr(LlamaModelParamsV2, "load_mtp")
    mp = LlamaModelParamsV2()
    assert hasattr(mp, "load_mtp")
    mp.load_mtp = True
    assert mp.load_mtp is True


def test_mtp_config_and_settings_schema():
    """Verify mtp_enabled setting is in default config and schema."""
    assert "mtp_enabled" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["mtp_enabled"] is True

    schema_field = next((f for f in CORE_FIELDS if f.key == "mtp_enabled"), None)
    assert schema_field is not None
    assert schema_field.group == "Engine"


def test_base_backend_and_engine_capability():
    """Verify supports_mtp capability defaults and exposure on Engine."""
    class DummyBackend(BaseBackend):
        @property
        def loaded(self) -> bool:
            return False
        def load(self): pass
        def unload(self): pass
        def chat_stream(self, *args, **kwargs):
            return iter([])
        def generate(self, *args, **kwargs): pass

    dummy = DummyBackend()
    assert dummy.supports_mtp is False

    with patch("localm.inference.engine.create_backend", return_value=dummy):
        engine = Engine("dummy-model")
        assert engine.supports_mtp is False


def test_llama_model_has_mtp_detection():
    """Verify GGUF metadata detection for MTP architectures."""
    mock_model = ctypes.c_void_p(1234)

    # 1. Direct native library check
    with patch.object(api, "load_lib") as mock_load_lib:
        mock_dll = MagicMock()
        mock_dll.llama_model_has_mtp.return_value = True
        mock_load_lib.return_value = mock_dll
        with patch.object(api, "_bind", return_value=lambda m: True):
            assert api.llama_model_has_mtp(mock_model) is True

    # 2. GGUF metadata check (DeepSeek nextn_predict_layers)
    with patch.object(api, "load_lib") as mock_load_lib, \
         patch.object(api, "has_model_meta_api", return_value=True):
        mock_dll = MagicMock(spec=[])  # no native llama_model_has_mtp
        mock_load_lib.return_value = mock_dll

        def fake_meta_val(model, key):
            if key == "general.architecture":
                return "deepseek2"
            if key == "deepseek2.nextn_predict_layers":
                return "1"
            return None

        with patch.object(api, "llama_model_meta_val_str", side_effect=fake_meta_val):
            assert api.llama_model_has_mtp(mock_model) is True

    # 3. GGUF metadata check (Qwen mtp_head_count)
    with patch.object(api, "load_lib") as mock_load_lib, \
         patch.object(api, "has_model_meta_api", return_value=True):
        mock_dll = MagicMock(spec=[])
        mock_load_lib.return_value = mock_dll

        def fake_meta_val_qwen(model, key):
            if key == "general.architecture":
                return "qwen2"
            if key == "qwen2.mtp_head_count":
                return "2"
            return None

        with patch.object(api, "llama_model_meta_val_str", side_effect=fake_meta_val_qwen):
            assert api.llama_model_has_mtp(mock_model) is True

    # 4. Standard non-MTP model
    with patch.object(api, "load_lib") as mock_load_lib, \
         patch.object(api, "has_model_meta_api", return_value=True):
        mock_dll = MagicMock(spec=[])
        mock_load_lib.return_value = mock_dll

        def fake_meta_val_none(model, key):
            if key == "general.architecture":
                return "llama"
            return None

        with patch.object(api, "llama_model_meta_val_str", side_effect=fake_meta_val_none):
            assert api.llama_model_has_mtp(mock_model) is False


def test_gguf_backend_supports_mtp():
    """Verify GgufBackend correctly reflects supports_mtp state."""
    backend = GgufBackend("test_model.gguf")
    assert backend.supports_mtp is False

    # Simulate load metadata with supports_mtp=True
    backend._loaded = True
    backend._supports_mtp = True
    assert backend.supports_mtp is True

    # Simulate load metadata with supports_mtp=False
    backend._supports_mtp = False
    assert backend.supports_mtp is False


def test_llama_cpp_mtp_speculative_verification_acceptance():
    """Verify speculative decode acceptance path in LlamaCpp._generate."""
    # Build a simulated LlamaCpp instance
    llm = object.__new__(LlamaCpp)
    llm._inference_lock = MagicMock()
    llm._gen_lock = MagicMock()
    llm._stop = MagicMock()
    llm._stop.is_set.return_value = False
    llm._model_ptr = ctypes.c_void_p(1)
    llm._ctx_ptr = ctypes.c_void_p(2)
    llm._mtp_ctx_ptr = ctypes.c_void_p(3)
    llm.supports_mtp = True
    llm._seed = 42
    llm._verbose = False
    llm._cached_tokens = []
    llm._ctx_capacity = 4096
    llm.close = MagicMock()

    tokenizer = MagicMock()
    tokenizer.is_eog.side_effect = lambda t: t == 999  # 999 is EOG
    llm._tokenizer = tokenizer

    llm._fit_generation_budget = lambda n_prompt, max_new: 4
    llm._can_reuse_kv = lambda needed: False
    llm._prefill_fresh_context = MagicMock()
    llm._create_batch = MagicMock(return_value=MagicMock())

    # Mock sampling and decoding:
    # 1. Base samples token 100
    # 2. MTP drafts token 101
    # 3. Base verification verifies token 101 -> MATCH! (accepted)
    # 4. Next base sample is 999 (EOG)
    sample_seq = [100, 101, 101, 999]
    sample_iter = iter(sample_seq)

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=MagicMock()):
        mock_api.llama_sampler_sample.side_effect = lambda s, ctx, idx: next(sample_iter)
        mock_api.llama_decode.return_value = 0

        tokens = list(llm._generate(
            prompt_tokens=[1, 2],
            max_new_tokens=4,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repeat_penalty=1.0,
        ))

        # Both token 100 and accepted draft token 101 are yielded
        assert 100 in tokens
        assert 101 in tokens
        assert 999 not in tokens


def test_llama_cpp_mtp_speculative_verification_rejection():
    """Verify speculative decode rejection and KV cache rollback path."""
    llm = object.__new__(LlamaCpp)
    llm._inference_lock = MagicMock()
    llm._gen_lock = MagicMock()
    llm._stop = MagicMock()
    llm._stop.is_set.return_value = False
    llm._model_ptr = ctypes.c_void_p(1)
    llm._ctx_ptr = ctypes.c_void_p(2)
    llm._mtp_ctx_ptr = ctypes.c_void_p(3)
    llm.supports_mtp = True
    llm._seed = 42
    llm._verbose = False
    llm._cached_tokens = []
    llm._ctx_capacity = 4096
    llm.close = MagicMock()

    tokenizer = MagicMock()
    tokenizer.is_eog.side_effect = lambda t: t == 999
    llm._tokenizer = tokenizer

    llm._fit_generation_budget = lambda n_prompt, max_new: 3
    llm._can_reuse_kv = lambda needed: False
    llm._prefill_fresh_context = MagicMock()
    llm._create_batch = MagicMock(return_value=MagicMock())

    # 1. Base samples token 200
    # 2. MTP drafts token 201
    # 3. Base verification returns 202 (MISMATCH -> REJECT 201)
    # 4. Next base sample: 999 (EOG)
    sample_seq = [200, 201, 202, 999]
    sample_iter = iter(sample_seq)

    with patch("localm.inference.backends.llamacpp.llama.api") as mock_api, \
         patch("localm.inference.backends.llamacpp.llama._build_sampler", return_value=MagicMock()):
        mock_api.llama_sampler_sample.side_effect = lambda s, ctx, idx: next(sample_iter)
        mock_api.llama_decode.return_value = 0

        tokens = list(llm._generate(
            prompt_tokens=[1, 2],
            max_new_tokens=3,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repeat_penalty=1.0,
        ))

        # Token 200 yielded, draft 201 rejected and rolled back
        assert 200 in tokens
        assert 201 not in tokens
        # Verify kv_cache rollback was called
        mock_api.llama_kv_cache_seq_rm.assert_called()
