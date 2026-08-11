# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline tests for scripts/check_llama_abi.py (the header-diff ABI verifier).

No network: the headers are embedded. Proves the verifier (a) agrees that
localm's named fields match the current upstream layout, (b) FAILS on a
mid-struct insertion, and (c) computes natural-alignment offsets correctly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_llama_abi.py"
_spec = importlib.util.spec_from_file_location("check_llama_abi", _PATH)
abichk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abichk)


# The structs as defined by upstream llama.cpp BEFORE the llama_model_params
# reorder (b9870 and older) - i.e. localm's V1 layout.
_GOOD_HEADER = """
struct llama_model_params {
    ggml_backend_dev_t * devices;
    const struct llama_model_tensor_buft_override * tensor_buft_overrides;
    int32_t n_gpu_layers;
    enum llama_split_mode split_mode;
    int32_t main_gpu;
    const float * tensor_split;
    llama_progress_callback progress_callback;
    void * progress_callback_user_data;
    const struct llama_model_kv_override * kv_overrides;
    bool vocab_only;
    bool use_mmap;
    bool use_direct_io;
    bool use_mlock;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
};
struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    int32_t n_threads;
    int32_t n_threads_batch;
    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;
    enum llama_attention_type attention_type;
    enum llama_flash_attn_type flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    enum ggml_type type_k;
    enum ggml_type type_v;
    ggml_abort_callback abort_callback;
    void * abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    struct llama_sampler_seq_config * samplers;
    size_t n_samplers;
    struct llama_context * ctx_other;
};
struct llama_batch {
    int32_t n_tokens;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
};
"""

# The same header AFTER upstream's in-place reorder of llama_model_params
# (lemonade b1307 / upstream b10105+) - localm's V2 layout. Note it is still 72
# bytes: load_mode is inserted at 24 and three booleans are replaced by it, which
# is precisely why a size check cannot detect this and an OFFSET check must.
_GOOD_HEADER_V2 = _GOOD_HEADER.replace(
    """    enum llama_split_mode split_mode;
    int32_t main_gpu;""",
    """    enum llama_split_mode split_mode;
    enum llama_load_mode load_mode;
    int32_t main_gpu;""",
).replace(
    """    bool use_mmap;
    bool use_direct_io;
    bool use_mlock;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;""",
    """    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
    bool load_mtp;""",
)

# A mid-struct insertion that shifts every later field (the dangerous drift).
_BAD_HEADER = _GOOD_HEADER.replace(
    "    uint32_t n_ctx;\n",
    "    uint32_t n_ctx;\n    int32_t injected_evil_field;\n",
)

# _GOOD_HEADER / _GOOD_HEADER_V2 both carry a context_params WITHOUT
# n_outputs_max_per_seq (context_params v1) - they only vary the
# model_params half, since that split predates the context_params one.
# This is context_params AFTER upstream's insertion (sometime between
# lemonade b1307 and ggml-org b10360) - localm's context_params V2 layout.
# Built off _GOOD_HEADER (model_params v1) since the two axes are
# independent; a real ggml-org b10360 header carries model_params v2 AND
# context_params v2 together, but nothing about the verifier assumes they
# move in lockstep, so testing them decoupled here is the stronger check.
_GOOD_HEADER_CTX_V2 = _GOOD_HEADER.replace(
    "    uint32_t n_outputs_max;\n",
    "    uint32_t n_outputs_max;\n    uint32_t n_outputs_max_per_seq;\n",
)


def test_embedded_headers_are_the_two_real_layouts():
    """Guards the fixtures themselves: if the V2 edit above stopped producing a
    genuinely different llama_model_params, every test below would silently
    check V1 twice and still pass."""
    assert abichk._header_model_params_layout(_GOOD_HEADER) == "v1"
    assert abichk._header_model_params_layout(_GOOD_HEADER_V2) == "v2"
    assert abichk._header_context_params_layout(_GOOD_HEADER) == "v1"
    assert abichk._header_context_params_layout(_GOOD_HEADER_CTX_V2) == "v2"


@pytest.mark.parametrize("struct", ["llama_model_params", "llama_context_params", "llama_batch"])
@pytest.mark.parametrize("header,layout", [
    (_GOOD_HEADER, "v1"), (_GOOD_HEADER_V2, "v2")])
def test_verifier_passes_on_matching_header(struct, header, layout):
    # Both fixtures are context_params v1; the model_params axis under test
    # varies via `layout`, independent of context - see _GOOD_HEADER_CTX_V2's
    # own dedicated coverage below for the context axis.
    assert abichk._check(struct, header, layout, "v1") == 0


def test_verifier_passes_on_matching_context_params_header():
    assert abichk._check("llama_context_params", _GOOD_HEADER_CTX_V2, "v1", "v2") == 0


@pytest.mark.parametrize("header,layout", [
    (_GOOD_HEADER, "v1"), (_GOOD_HEADER_V2, "v2")])
def test_verifier_fails_on_midstruct_insertion(header, layout):
    # The injected field shifts n_batch onward -> many offset mismatches.
    # model_layout is irrelevant here (_check ignores it for
    # llama_context_params), context_layout is "v1" since these fixtures are.
    bad = header.replace(
        "    uint32_t n_ctx;\n",
        "    uint32_t n_ctx;\n    int32_t injected_evil_field;\n")
    assert abichk._check("llama_context_params", bad, layout, "v1") > 0


def test_verifier_fails_when_the_wrong_model_params_layout_is_selected():
    """The upgrade's core hazard, as the offline verifier sees it: a V2 header
    checked against the V1 class (what a stale binding does) must FAIL, and vice
    versa. If either direction passed, the two-layout split would be cosmetic."""
    assert abichk._check("llama_model_params", _GOOD_HEADER_V2, "v1", "v1") > 0
    assert abichk._check("llama_model_params", _GOOD_HEADER, "v2", "v1") > 0


def test_verifier_fails_when_the_wrong_context_params_layout_is_selected():
    """Same hazard, the newer axis: a context_params v2 header checked against
    the v1 class (or vice versa) must FAIL - this is the exact check that
    would have caught the n_outputs_max_per_seq insertion before it ever
    reached a user, had the verifier been run against a current header."""
    assert abichk._check("llama_context_params", _GOOD_HEADER_CTX_V2, "v1", "v1") > 0
    assert abichk._check("llama_context_params", _GOOD_HEADER, "v1", "v2") > 0


def test_layout_natural_alignment():
    fields = abichk._parse_fields("uint32_t a; uint64_t b; bool c;")
    layout, size = abichk._layout(fields)
    assert layout == [("a", 0, 4), ("b", 8, 8), ("c", 16, 1)]
    assert size == 24   # aligned up to the 8-byte max alignment


def test_field_sizes():
    assert abichk._field_size("int32_t ") == 4
    assert abichk._field_size("enum llama_pooling_type ") == 4
    assert abichk._field_size("size_t ") == 8
    assert abichk._field_size("void * ") == 8
    assert abichk._field_size("ggml_abort_callback ") == 8   # fn-pointer typedef
    assert abichk._field_size("bool ") == 1
    assert abichk._field_size("struct llama_context * ") == 8
