# SPDX-License-Identifier: AGPL-3.0-or-later
"""Byte-exact synthetic GGUF headers for tests: typed metadata values and a tensor
list, enough for the metadata probe and the tensor-name scan to read."""

from __future__ import annotations

import struct

from localm.model_manager import gguf as _gguf

STRING, UINT32, INT32, BOOL, ARRAY = 8, 4, 5, 7, 9


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, value) -> bytes:
    out = _s(key) + struct.pack("<I", kind)
    if kind == STRING:
        out += _s(value)
    elif kind == UINT32:
        out += struct.pack("<I", value)
    elif kind == INT32:
        out += struct.pack("<i", value)
    elif kind == BOOL:
        out += struct.pack("<?", value)
    elif kind == ARRAY:
        out += struct.pack("<I", STRING) + struct.pack("<Q", len(value))
        out += b"".join(_s(v) for v in value)
    else:
        raise ValueError(f"unsupported kind {kind}")
    return out


def build_gguf(architecture: str, kvs=(), tensors=()) -> bytes:
    """A GGUF v3 header. *kvs* is ``[(key, kind, value), ...]`` written after
    ``general.architecture``; *tensors* is a list of tensor names (each a
    one-dimensional F32 tensor of 4 elements). Padded to the size floor the model
    manager requires of a real file."""
    entries = [("general.architecture", STRING, architecture), *kvs]
    buf = bytearray(b"GGUF")
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", len(entries))
    for key, kind, value in entries:
        buf += _kv(key, kind, value)
    for index, name in enumerate(tensors):
        buf += _s(name)
        buf += struct.pack("<I", 1)             # n_dims
        buf += struct.pack("<Q", 4)             # dims[0]
        buf += struct.pack("<I", 0)             # ggml type F32
        buf += struct.pack("<Q", index * 16)    # offset
    floor = _gguf._GGUF_MIN_BYTES
    if len(buf) < floor:
        buf += b"\x00" * (floor - len(buf))
    return bytes(buf)


def reranker_bytes(architecture: str = "bert", *, rank_key: bool = False,
                   labels=None, tensors=("cls.weight", "cls.output.weight")) -> bytes:
    """A reranker GGUF: a classification head and, optionally, the RANK pooling
    and label keys community conversions often leave out."""
    kvs = []
    if rank_key:
        kvs.append((f"{architecture}.pooling_type", UINT32, 4))
    if labels is not None:
        kvs.append((f"{architecture}.classifier.output_labels", ARRAY, list(labels)))
    return build_gguf(architecture, kvs, tensors)


def embedding_bytes(architecture: str = "bert", *, pooling: int | None = None,
                    tensors=("token_embd.weight",)) -> bytes:
    """A plain embedding GGUF: no classification head."""
    kvs = [] if pooling is None else [(f"{architecture}.pooling_type", UINT32, pooling)]
    return build_gguf(architecture, kvs, tensors)
