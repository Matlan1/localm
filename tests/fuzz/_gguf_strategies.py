# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hypothesis strategies that emit GGUF headers: structurally plausible ones
(real key names, coherent architecture prefixes) with the lies a hostile file
tells layered on top - declared counts and lengths that disagree with the
bytes, wrong value types, truncation and byte flips."""
from __future__ import annotations

import struct

from hypothesis import strategies as st

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I",
    T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q",
    T_FLOAT64: "<d",
}

_ARCHITECTURES = ["llama", "qwen3moe", "qwen3next", "lfm2", "granite", "clip", "bert",
                  "nomic-bert", "gemma3", "mamba", "t5", "deepseek2", "glm4moe", "x"]

_SHAPE_SUFFIXES = [
    "block_count", "embedding_length", "context_length", "expert_count",
    "expert_used_count", "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length", "nextn_predict_layers",
    "pooling_type", "ssm.state_size", "ssm.conv_kernel", "ssm.inner_size",
    "shortconv.l_cache", "full_attention_interval", "feed_forward_length",
    "rope.dimension_count", "vocab_size", "expert_feed_forward_length",
]

_BARE_KEYS = [
    "general.name", "general.alignment", "general.file_type", "general.type",
    "tokenizer.chat_template", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
    "tokenizer.ggml.model", "split.count", "split.no", "split.tensors.count",
    "clip.projector_type", "clip.vision.projector_type", "clip.has_vision_encoder",
    "general.architecture",
]

_TENSOR_NAMES = [
    "token_embd.weight", "output.weight", "output_norm.weight", "blk.0.attn_k.weight",
    "blk.0.attn_v.weight", "blk.0.attn_q.weight", "blk.0.attn_norm.weight",
    "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_gate_exps.weight",
    "blk.3.ssm_a", "mm.0.weight", "mm.2.weight", "mm.model.mb_block.1.block.0.weight",
    "mm.model.peg.0.weight", "v.patch_embd.weight", "blk.0.nextn.eh_proj.weight",
]

_EDGE_COUNTS = [0, 1, 2, 255, 256, 65535, 2**31 - 1, 2**31, 2**32 - 1, 2**32,
                2**40, 2**62, 2**63, 2**64 - 1]

_utf8_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), max_size=24)


def lstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _int_for(fmt: str) -> st.SearchStrategy:
    size = struct.calcsize(fmt)
    bits = size * 8
    if fmt.lower() == fmt and fmt not in ("<?",):
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        lo, hi = 0, 2 ** bits - 1
    return st.one_of(st.sampled_from([lo, hi, 0, 1, 2, 4, 8, 32, 64, 128]).filter(
        lambda v: lo <= v <= hi), st.integers(lo, hi))


@st.composite
def scalar_value(draw, vtype=None):
    if vtype is None:
        vtype = draw(st.sampled_from(sorted(_FIXED)))
    fmt = _FIXED[vtype]
    if fmt in ("<f", "<d"):
        v = draw(st.floats(allow_nan=True, allow_infinity=True, width=32 if fmt == "<f" else 64))
    elif fmt == "<?":
        v = draw(st.booleans())
    else:
        v = draw(_int_for(fmt))
    return vtype, struct.pack(fmt, v)


@st.composite
def string_value(draw):
    text = draw(_utf8_text)
    raw = text.encode("utf-8")
    if draw(st.integers(0, 9)) == 0:
        declared = draw(st.sampled_from([0, len(raw) + 1, len(raw) + 4096, 2**20, 2**32,
                                         2**63, 2**64 - 1]))
        return T_STRING, struct.pack("<Q", declared) + raw
    if draw(st.integers(0, 14)) == 0:
        return T_STRING, struct.pack("<Q", 2) + draw(st.binary(min_size=2, max_size=2))
    return T_STRING, struct.pack("<Q", len(raw)) + raw


@st.composite
def array_value(draw, depth=0):
    elem_types = sorted(_FIXED) + [T_STRING, 13, 200]
    if depth < 1:
        elem_types.append(T_ARRAY)
    elem = draw(st.sampled_from(elem_types))
    n = draw(st.integers(0, 6))
    body = b""
    for _ in range(n):
        if elem in _FIXED:
            body += draw(scalar_value(elem))[1]
        elif elem == T_STRING:
            body += draw(string_value())[1]
        elif elem == T_ARRAY:
            body += draw(array_value(depth + 1))[1]
        else:
            body += draw(st.binary(max_size=4))
    declared = n
    if draw(st.integers(0, 4)) == 0:
        declared = draw(st.sampled_from(_EDGE_COUNTS))
    return T_ARRAY, struct.pack("<IQ", elem, declared) + body


@st.composite
def typed_value(draw):
    kind = draw(st.sampled_from(["scalar", "scalar", "string", "array", "garbage"]))
    if kind == "scalar":
        return draw(scalar_value())
    if kind == "string":
        return draw(string_value())
    if kind == "array":
        return draw(array_value())
    vtype = draw(st.sampled_from([0, 4, 8, 9, 13, 99, 2**32 - 1]))
    return vtype, draw(st.binary(max_size=16))


@st.composite
def kv_entries(draw):
    arch = draw(st.sampled_from(_ARCHITECTURES))
    entries = []
    if draw(st.integers(0, 9)) != 0:
        entries.append(("general.architecture", T_STRING, lstr(arch)))
    n = draw(st.integers(0, 14))
    for _ in range(n):
        kind = draw(st.integers(0, 9))
        if kind <= 5:
            key = f"{arch}.{draw(st.sampled_from(_SHAPE_SUFFIXES))}"
        elif kind <= 8:
            key = draw(st.sampled_from(_BARE_KEYS))
        else:
            key = draw(_utf8_text)
        if draw(st.integers(0, 3)) == 0:
            vtype, body = draw(typed_value())
        elif key == "general.architecture" or key.endswith(("chat_template", ".pre", ".name",
                                                          "projector_type", ".model")):
            vtype, body = draw(string_value())
        elif key.endswith("tokens"):
            vtype, body = T_ARRAY, struct.pack("<IQ", T_STRING, 3) + b"".join(
                lstr(t) for t in ("a", "b", "c"))
        elif "head_count_kv" in key and draw(st.booleans()):
            count = draw(st.integers(0, 40))
            items = [draw(st.integers(0, 16)) for _ in range(count)]
            vtype, body = T_ARRAY, struct.pack("<IQ", T_INT32, count) + b"".join(
                struct.pack("<i", v) for v in items)
        else:
            vtype, body = draw(scalar_value(draw(st.sampled_from([T_UINT32, T_UINT32, T_INT32,
                                                                  T_UINT64, T_UINT16]))))
        entries.append((key, vtype, body))
    return entries


@st.composite
def tensor_infos(draw):
    n = draw(st.integers(0, 10))
    out = []
    for _ in range(n):
        name = draw(st.sampled_from(_TENSOR_NAMES) | _utf8_text)
        n_dims = draw(st.sampled_from([0, 1, 2, 2, 3, 4, 5, 8, 9, 64, 2**32 - 1]))
        real_dims = min(n_dims, 6)
        dims = [draw(st.sampled_from([0, 1, 2, 4096, 2**32, 2**63, 2**64 - 1])
                     | st.integers(0, 70000)) for _ in range(real_dims)]
        ggml_type = draw(st.sampled_from([0, 1, 2, 8, 12, 30, 34, 35, 36, 200, 2**32 - 1]))
        offset = draw(st.sampled_from([0, 32, 2**32, 2**63, 2**64 - 1]) | st.integers(0, 2**20))
        out.append((name, n_dims, dims, ggml_type, offset))
    return out


_SMALL = st.sampled_from([0, 1, 2, 3, 4, 8, 16, 32, 40, 64, 128, 4096, 2 ** 31 - 1, 2 ** 32 - 1])
_DIMS = st.sampled_from([0, 1, 2, 3, 8, 64, 768, 4096, 2 ** 32, 2 ** 63])


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


@st.composite
def coherent_parts(draw):
    """KV entries and tensor infos shaped like a real model file (a text model,
    a hybrid, a MoE, a clip mmproj), with the numbers drawn from hostile
    edges, so the readers' arithmetic past the structural checks gets run."""
    kind = draw(st.sampled_from(["dense", "hybrid", "moe", "mmproj", "mmproj", "split"]))
    arch = "clip" if kind == "mmproj" else draw(
        st.sampled_from(["llama", "qwen3next", "lfm2", "granite", "qwen3moe", "glm4moe"]))
    kvs = [("general.architecture", T_STRING, lstr(arch))]
    tensors = []

    def uint(key, value=None):
        v = draw(_SMALL) if value is None else value
        kvs.append((key, T_UINT32, _u32(v)))

    if kind == "mmproj":
        ptype = draw(st.sampled_from(["mlp", "ldp", "ldpv2", "resampler", "merger", None]))
        if ptype is not None:
            kvs.append((draw(st.sampled_from(["clip.projector_type",
                                              "clip.vision.projector_type"])),
                        T_STRING, lstr(ptype)))
        if draw(st.booleans()):
            uint("clip.vision.projection_dim")
        for name in draw(st.lists(st.sampled_from([
                "mm.0.weight", "mm.2.weight", "mm.3.weight", "mm.3.bias",
                "mm.model.mb_block.1.block.2.1.bias", "mm.model.peg.0.bias",
                "mm.model.mb_block.0.weight", "v.blk.0.attn_k.weight"]), max_size=6)):
            n_dims = draw(st.integers(1, 3))
            tensors.append((name, n_dims, [draw(_DIMS) for _ in range(n_dims)], 0, 0))
        return kvs, tensors

    n_layers = draw(st.integers(0, 6))
    uint(f"{arch}.block_count", n_layers)
    for suffix in ("embedding_length", "attention.head_count", "attention.key_length",
                   "attention.value_length", "context_length", "expert_count",
                   "expert_used_count", "nextn_predict_layers", "ssm.conv_kernel",
                   "ssm.state_size", "ssm.inner_size", "ssm.group_count",
                   "shortconv.l_cache", "full_attention_interval"):
        if draw(st.booleans()):
            uint(f"{arch}.{suffix}")
    if draw(st.booleans()):
        count = n_layers if draw(st.booleans()) else draw(st.integers(0, 8))
        kvs.append((f"{arch}.attention.head_count_kv", T_ARRAY,
                    struct.pack("<IQ", T_INT32, count)
                    + b"".join(struct.pack("<i", draw(st.integers(-2, 16)))
                               for _ in range(count))))
    else:
        uint(f"{arch}.attention.head_count_kv")
    if kind == "split":
        uint("split.count", draw(st.integers(0, 5)))
        uint("split.no", draw(st.integers(0, 5)))
    kvs.append(("tokenizer.ggml.tokens", T_ARRAY, struct.pack("<IQ", T_STRING, 2)
                + lstr("a") + lstr("b")))
    offset = 0
    for layer in range(n_layers):
        for stem in draw(st.lists(st.sampled_from([
                "attn_k", "attn_v", "attn_q", "attn_norm", "ffn_up_exps", "ffn_down_exps",
                "ffn_gate_exps", "ssm_a", "ssm_conv1d", "nextn.eh_proj", "shortconv.conv"]),
                max_size=5, unique=True)):
            tensors.append((f"blk.{layer}.{stem}.weight", 2, [draw(_DIMS), draw(_DIMS)], 0,
                            offset))
            offset = (offset + draw(st.sampled_from([0, 32, 4096, 2 ** 31, 2 ** 63]))) % 2 ** 64
    for name in ("token_embd.weight", "output.weight", "output_norm.weight"):
        if draw(st.booleans()):
            tensors.append((name, 2, [draw(_DIMS), draw(_DIMS)], 0, offset))
    return kvs, tensors


def _tensor_bytes(info) -> bytes:
    name, n_dims, dims, ggml_type, offset = info
    return (lstr(name) + struct.pack("<I", n_dims)
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<IQ", ggml_type, offset))


@st.composite
def gguf_bytes(draw):
    """A GGUF-shaped byte string, possibly lying about counts, then possibly
    truncated or byte-flipped."""
    kvs, tensors = draw(st.one_of(st.tuples(kv_entries(), tensor_infos()),
                                  coherent_parts(), coherent_parts()))
    version = draw(st.sampled_from([3, 3, 3, 3, 3, 2, 1, 0, 4, 2**32 - 1]))
    kv_count = len(kvs)
    tensor_count = len(tensors)
    if draw(st.integers(0, 5)) == 0:
        kv_count = draw(st.sampled_from(_EDGE_COUNTS))
    if draw(st.integers(0, 5)) == 0:
        tensor_count = draw(st.sampled_from(_EDGE_COUNTS))
    magic = b"GGUF" if draw(st.integers(0, 19)) else draw(st.binary(min_size=4, max_size=4))
    data = bytearray(magic + struct.pack("<IQQ", version, tensor_count, kv_count))
    for key, vtype, body in kvs:
        data += lstr(key) + struct.pack("<I", vtype) + body
    for info in tensors:
        data += _tensor_bytes(info)
    data += draw(st.binary(max_size=64))
    if draw(st.integers(0, 3)) == 0 and data:
        del data[draw(st.integers(0, len(data) - 1)):]
    for _ in range(draw(st.integers(0, 3)) if draw(st.integers(0, 3)) == 0 else 0):
        if data:
            data[draw(st.integers(0, len(data) - 1))] = draw(st.integers(0, 255))
    return bytes(data)


raw_gguf_like = st.binary(max_size=256).map(lambda b: b"GGUF" + b)
