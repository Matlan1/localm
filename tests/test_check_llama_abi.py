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


# The structs as defined by upstream llama.cpp before the llama_model_params
# reorder - localm's V1 layout.
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

# The same header after upstream's in-place reorder of llama_model_params -
# localm's V2 layout. Still 72 bytes: load_mode is inserted at offset 24 and
# replaces three booleans, so only an OFFSET check sees the change.
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

# A mid-struct insertion that shifts every later field.
_BAD_HEADER = _GOOD_HEADER.replace(
    "    uint32_t n_ctx;\n",
    "    uint32_t n_ctx;\n    int32_t injected_evil_field;\n",
)

# _GOOD_HEADER / _GOOD_HEADER_V2 both carry a context_params WITHOUT
# n_outputs_max_per_seq (context_params v1); they vary only the model_params half.
# This is context_params AFTER upstream's insertion - localm's context_params V2
# layout. Built off _GOOD_HEADER (model_params v1); the two axes are independent.
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
    # Both fixtures are context_params v1; the model_params axis under test varies
    # via `layout`, independent of context.
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


# --------------------------------------------------------------------------- #
#  Enum DOMAIN checking
#
#  A layout check reads WHERE a field sits and is structurally blind to WHICH
#  VALUES are legal in it. These tests pin the detector for that class.
#
#  Everything below is driven off a SYNTHESISED header, never a real upstream tag.
# --------------------------------------------------------------------------- #

# RANK is omitted from llama_pooling_type here, so this fixture's clean state
# produces ZERO additive notes, and the additive assertions below can name the
# injected member exactly.
_ENUM_BLOCKS = """
enum llama_load_mode {
    LLAMA_LOAD_MODE_AUTO       = -1,
    LLAMA_LOAD_MODE_NONE       = 0,
    LLAMA_LOAD_MODE_MMAP       = 1,
    LLAMA_LOAD_MODE_MLOCK      = 2,
    LLAMA_LOAD_MODE_MMAP_MLOCK = 3,
    LLAMA_LOAD_MODE_DIRECT_IO  = 4,
};
enum llama_pooling_type {
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE = 0,
    LLAMA_POOLING_TYPE_MEAN = 1,
    LLAMA_POOLING_TYPE_CLS  = 2,
    LLAMA_POOLING_TYPE_LAST = 3,
};
"""

# _GOOD_HEADER_V2 carries llama_model_params.load_mode and (via _GOOD_HEADER)
# llama_context_params.pooling_type, which are the two gate fields.
_ENUM_HEADER = _GOOD_HEADER_V2 + _ENUM_BLOCKS

# (a) upstream ADDS a member localm does not bind. Additive.
_ENUM_ADDED = _ENUM_HEADER.replace(
    "    LLAMA_LOAD_MODE_DIRECT_IO  = 4,\n",
    "    LLAMA_LOAD_MODE_DIRECT_IO  = 4,\n    LLAMA_LOAD_MODE_TELEPORT   = 5,\n")

# (b) upstream CHANGES the value of a member localm already binds. Corrupting.
_ENUM_CHANGED = _ENUM_HEADER.replace(
    "    LLAMA_LOAD_MODE_MMAP       = 1,\n",
    "    LLAMA_LOAD_MODE_MMAP       = 7,\n")

# The header as it stood at the v2 pin b10360, which predates AUTO while localm
# binds it. This must NOT fail.
_ENUM_NO_AUTO = _ENUM_HEADER.replace(
    "    LLAMA_LOAD_MODE_AUTO       = -1,\n", "")

_LOAD_MODE = next(b for b in abichk._ENUM_BINDINGS if b.c_enum == "llama_load_mode")
_POOLING = next(b for b in abichk._ENUM_BINDINGS if b.c_enum == "llama_pooling_type")


def _run_enum(binding, header):
    """(problem count, additive notes) for one binding against one header."""
    notes = []
    return abichk._check_enum(binding, header, notes), notes


def test_enum_fixture_edits_actually_took():
    """Guards the FAULT INJECTORS, not the code. str.replace silently no-ops on a
    pattern that does not match, and a fault that never fired is indistinguishable
    from a checker that correctly found nothing to report: both give a green run.
    Every test below is worthless if these strings are equal."""
    assert _ENUM_ADDED != _ENUM_HEADER
    assert _ENUM_CHANGED != _ENUM_HEADER
    assert _ENUM_NO_AUTO != _ENUM_HEADER
    assert "LLAMA_LOAD_MODE_TELEPORT" in _ENUM_ADDED
    assert "LLAMA_LOAD_MODE_MMAP       = 7" in _ENUM_CHANGED
    assert "LLAMA_LOAD_MODE_AUTO" not in _ENUM_NO_AUTO
    # The clean fixture produces no additive notes of its own.
    assert _run_enum(_LOAD_MODE, _ENUM_HEADER) == (0, [])


def test_added_member_and_changed_value_are_distinct_outcomes():
    """THE ORACLE. Collapsing these two into one outcome is the defect, not the
    fix: hard-failing on an addition trains people to widen localm's accept-sets
    to silence the gate, which destroys the misaligned-read tripwire that reads
    them; passing a changed value through lets a number quietly change meaning."""
    added_problems, added_notes = _run_enum(_LOAD_MODE, _ENUM_ADDED)
    changed_problems, changed_notes = _run_enum(_LOAD_MODE, _ENUM_CHANGED)

    # Additive: reported loudly, exit outcome unchanged.
    assert added_problems == 0
    assert added_notes == ["llama_load_mode.LLAMA_LOAD_MODE_TELEPORT = 5"]

    # Changed value for a bound name: hard fail, and NOT routed to the quiet
    # additive channel.
    assert changed_problems > 0
    assert changed_notes == []

    # The two inputs must not produce the same exit outcome.
    assert (added_problems > 0) != (changed_problems > 0)


def test_changed_value_names_both_numbers(capsys):
    """A bare "mismatch" sends the reader to the header to work out which side
    moved. The report has to carry both values to be actionable."""
    _run_enum(_LOAD_MODE, _ENUM_CHANGED)
    out = capsys.readouterr().out
    assert "LLAMA_LOAD_MODE_MMAP" in out
    assert "localm binds 1" in out and "upstream says 7" in out


def test_header_predating_a_bound_member_is_a_note_not_a_failure():
    """localm binds AUTO = -1 and the pinned ref predates that member, so the
    check reports a note rather than failing."""
    problems, notes = _run_enum(_LOAD_MODE, _ENUM_NO_AUTO)
    assert problems == 0
    assert notes == []      # not additive either: localm binds it, upstream lacks it


def test_enum_absent_with_its_field_absent_is_skipped():
    """_GOOD_HEADER is pre-reorder: no llama_model_params.load_mode and no
    llama_load_mode. That header predates the feature and is not drift."""
    assert _run_enum(_LOAD_MODE, _GOOD_HEADER) == (0, [])


def test_enum_absent_while_its_field_is_present_fails():
    """The discriminator a bare absence check cannot make. _GOOD_HEADER_V2 has
    "enum llama_load_mode load_mode;" in the struct but no enum definition, so
    the domain is UNVERIFIED, which must not read the same as verified."""
    problems, notes = _run_enum(_LOAD_MODE, _GOOD_HEADER_V2)
    assert problems > 0
    assert notes == []


def test_a_localm_policy_constant_is_not_reported_as_a_binding():
    """embedder._POOLING_DEFAULT is localm's own choice of MEAN, not an upstream
    enumerator. An earlier prefix-scanning version of the registry reported
    "localm binds LLAMA_POOLING_TYPE_DEFAULT", which is simply false. A verifier
    that invents a binding is worse than one with a narrower reach."""
    problems, notes = _run_enum(_POOLING, _ENUM_HEADER)
    assert problems == 0
    assert notes == []


def test_registry_names_constants_that_still_exist():
    """If someone renames a constant in localm, the registry stops comparing it
    and every remaining member still reads "ok" - a check that quietly shrank."""
    for binding in abichk._ENUM_BINDINGS:
        bound, missing, _ = abichk._localm_enum_binding(binding)
        assert missing == [], f"{binding.c_enum}: {binding.module} lost {missing}"
        assert bound, f"{binding.c_enum}: nothing bound"


def test_registry_that_forgot_a_real_member_fails():
    """The self-audit that keeps the explicit member list from going stale: a
    constant localm defines under the prefix, which upstream also defines as a
    member of this enum, is a binding the registry forgot."""
    stale = _LOAD_MODE._replace(
        members=tuple(m for m in _LOAD_MODE.members if m != "MMAP"))
    problems, _ = _run_enum(stale, _ENUM_HEADER)
    assert problems > 0


def test_registry_missing_constant_fails_rather_than_silently_shrinking():
    bogus = _LOAD_MODE._replace(members=_LOAD_MODE.members + ("NO_SUCH_MEMBER",))
    problems, _ = _run_enum(bogus, _ENUM_HEADER)
    assert problems > 0


def test_parse_enum_members_implicit_and_explicit():
    members, unreadable = abichk._parse_enum_members("A, B, C")
    assert members == {"A": 0, "B": 1, "C": 2} and not unreadable
    members, unreadable = abichk._parse_enum_members("A = 5, B, C = -1, D")
    assert members == {"A": 5, "B": 6, "C": -1, "D": 0} and not unreadable
    members, _ = abichk._parse_enum_members("A = 0x10, B")
    assert members == {"A": 16, "B": 17}
    members, _ = abichk._parse_enum_members("A = -1, B = 0, C = 1,")
    assert members == {"A": -1, "B": 0, "C": 1}     # trailing comma


def test_parse_enum_members_does_not_guess_after_an_unreadable_value():
    """A non-literal value poisons every implicit member after it. Guessing one
    would be indistinguishable from a read one and could manufacture a false ok
    on the exact comparison this file exists to make."""
    members, unreadable = abichk._parse_enum_members("A = SOME_MACRO, B, C = 9")
    assert "A" in unreadable and "B" in unreadable
    assert "A" not in members and "B" not in members
    assert members == {"C": 9}      # an explicit value after the poison still reads


def test_comments_do_not_leak_into_enum_values():
    members, unreadable = abichk._parse_enum_members(
        "A = 0, // memory map the model\n    B = 1, /* both */\n")
    assert members == {"A": 0, "B": 1} and not unreadable
