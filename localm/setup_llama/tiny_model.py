# SPDX-License-Identifier: AGPL-3.0-or-later
"""A llama model of a few kilobytes with made-up weights, to prove that a provisioned
runtime loads a model and runs it. Its answers mean nothing.
"""

from __future__ import annotations

import struct
from pathlib import Path

_U32, _F32, _STRING, _ARRAY, _I32 = 4, 6, 8, 9, 5
_EMBD, _FFN, _HEADS, _ALIGN = 32, 64, 2, 32
_SPECIAL = ("<unk>", "<s>", "</s>")


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, value) -> bytes:
    head = _string(key) + struct.pack("<I", kind)
    if kind == _STRING:
        return head + _string(value)
    if kind == _ARRAY:
        inner, items = value
        body = (b"".join(_string(v) for v in items) if inner == _STRING else
                struct.pack(f"<{len(items)}{'f' if inner == _F32 else 'i'}", *items))
        return head + struct.pack("<IQ", inner, len(items)) + body
    return head + struct.pack("<f" if kind == _F32 else "<I", value)


def _values(n: int, seed: int) -> bytes:
    out = []
    for _ in range(n):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((seed / 0x7FFFFFFF - 0.5) * 0.2)
    return struct.pack(f"<{n}f", *out)


def write(path: Path) -> Path:
    """Write the model to *path* (GGUF) and return *path*. Raises OSError."""
    tokens = [*_SPECIAL, *(f"<0x{b:02X}>" for b in range(256))]
    kinds = [2, 3, 3] + [6] * 256
    vocab = len(tokens)
    kvs = [
        _kv("general.architecture", _STRING, "llama"),
        _kv("general.name", _STRING, "localm runtime check"),
        _kv("llama.context_length", _U32, 64),
        _kv("llama.embedding_length", _U32, _EMBD),
        _kv("llama.block_count", _U32, 1),
        _kv("llama.feed_forward_length", _U32, _FFN),
        _kv("llama.attention.head_count", _U32, _HEADS),
        _kv("llama.attention.head_count_kv", _U32, _HEADS),
        _kv("llama.attention.layer_norm_rms_epsilon", _F32, 1e-5),
        _kv("llama.rope.dimension_count", _U32, _EMBD // _HEADS),
        _kv("llama.vocab_size", _U32, vocab),
        _kv("tokenizer.ggml.model", _STRING, "llama"),
        _kv("tokenizer.ggml.tokens", _ARRAY, (_STRING, tokens)),
        _kv("tokenizer.ggml.scores", _ARRAY, (_F32, [0.0] * vocab)),
        _kv("tokenizer.ggml.token_type", _ARRAY, (_I32, kinds)),
        _kv("tokenizer.ggml.unknown_token_id", _U32, 0),
        _kv("tokenizer.ggml.bos_token_id", _U32, 1),
        _kv("tokenizer.ggml.eos_token_id", _U32, 2),
    ]
    shapes = {"token_embd.weight": (_EMBD, vocab), "output.weight": (_EMBD, vocab),
              "output_norm.weight": (_EMBD,)}
    for name, shape in (("attn_q", (_EMBD, _EMBD)), ("attn_k", (_EMBD, _EMBD)),
                        ("attn_v", (_EMBD, _EMBD)), ("attn_output", (_EMBD, _EMBD)),
                        ("attn_norm", (_EMBD,)), ("ffn_norm", (_EMBD,)),
                        ("ffn_gate", (_EMBD, _FFN)), ("ffn_down", (_FFN, _EMBD)),
                        ("ffn_up", (_EMBD, _FFN))):
        shapes[f"blk.0.{name}.weight"] = shape
    infos, data = [], b""
    for seed, (name, shape) in enumerate(shapes.items(), 1):
        count = shape[0] * (shape[1] if len(shape) > 1 else 1)
        infos.append(_string(name) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, 0,
                                                 len(data)))
        data += (struct.pack("<f", 1.0) * count if "norm" in name else _values(count, seed))
        data += b"\0" * (-len(data) % _ALIGN)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(shapes), len(kvs)) + b"".join(kvs) + b"".join(infos)
    path = Path(path)
    path.write_bytes(head + b"\0" * (-len(head) % _ALIGN) + data)
    return path
