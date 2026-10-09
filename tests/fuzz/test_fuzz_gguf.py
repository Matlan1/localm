# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the GGUF header readers in ``localm.model_manager.gguf``.

A GGUF file reaches these readers from a download, a folder scan or a drag and
drop, so every field is attacker-controlled. The contract the callers rely on:
the readers answer "no signal" for a file they cannot parse and never raise,
never hang, and never allocate in proportion to a count the file declares."""
from __future__ import annotations

import itertools
import struct
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, strategies as st  # noqa: E402

from localm.model_manager import gguf  # noqa: E402
from tests.fuzz import _bounds  # noqa: E402
from tests.fuzz._gguf_strategies import gguf_bytes, raw_gguf_like  # noqa: E402

_counter = itertools.count()

NEVER_RAISE = {
    "gguf_kv_bytes_per_token": gguf.gguf_kv_bytes_per_token,
    "gguf_nextn_predict_layers": gguf.gguf_nextn_predict_layers,
    "gguf_mtp_draft_kv_bytes_per_token":
        lambda p: gguf.gguf_mtp_draft_kv_bytes_per_token(p, 3),
    "gguf_expert_count": gguf.gguf_expert_count,
    "gguf_expert_counts": gguf.gguf_expert_counts,
    "gguf_moe_pinned_expert_bytes": lambda p: gguf.gguf_moe_pinned_expert_bytes(p, 2),
    "gguf_block_bytes": gguf.gguf_block_bytes,
    "gguf_moe_expert_bytes_by_layer": gguf.gguf_moe_expert_bytes_by_layer,
    "gguf_input_layer_bytes": gguf.gguf_input_layer_bytes,
    "gguf_recurrent_state_bytes": gguf.gguf_recurrent_state_bytes,
    "gguf_split_layout": gguf.gguf_split_layout,
    "_gguf_declared_min_size": gguf._gguf_declared_min_size,
    "_gguf_metadata_probe": gguf._gguf_metadata_probe,
    "gguf_pretokenizer": gguf.gguf_pretokenizer,
    "gguf_embedding_signal": gguf.gguf_embedding_signal,
    "gguf_architecture": gguf.gguf_architecture,
    "gguf_is_mmproj": gguf.gguf_is_mmproj,
    "gguf_mmproj_inferred_projector_type": gguf.gguf_mmproj_inferred_projector_type,
    "_gguf_attending_layer_count": lambda p: gguf._gguf_attending_layer_count(p, 8),
    "_has_gguf_magic": gguf._has_gguf_magic,
}

UNDOCUMENTED = {
    "gguf_n_embd": gguf.gguf_n_embd,
    "gguf_registry_metadata": gguf.gguf_registry_metadata,
    "gguf_tool_use_signal": gguf.gguf_tool_use_signal,
    "gguf_context_length": gguf.gguf_context_length,
    "gguf_capability_metadata": gguf.gguf_capability_metadata,
    "_gguf_capability_probe": gguf._gguf_capability_probe,
}


def _write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / f"f{next(_counter)}.gguf"
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("name", sorted({**NEVER_RAISE, **UNDOCUMENTED}))
@given(data=st.one_of(gguf_bytes(), raw_gguf_like, st.binary(max_size=64)))
def test_reader_never_raises_and_returns_promptly(name, data, tmp_path):
    fn = {**NEVER_RAISE, **UNDOCUMENTED}[name]
    path = _write(tmp_path, data)
    try:
        _bounds.returns_within(fn, path)
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("name", [
    "gguf_kv_bytes_per_token", "gguf_expert_counts", "gguf_block_bytes",
    "gguf_input_layer_bytes", "gguf_moe_expert_bytes_by_layer", "_gguf_metadata_probe",
    "gguf_n_embd", "gguf_recurrent_state_bytes", "gguf_split_layout",
])
@given(data=gguf_bytes())
def test_reader_allocation_is_not_proportional_to_declared_counts(name, data, tmp_path):
    fn = {**NEVER_RAISE, **UNDOCUMENTED}[name]
    path = _write(tmp_path, data)
    try:
        _, peak = _bounds.peak_allocation(fn, path)
    finally:
        path.unlink(missing_ok=True)
    assert peak < _bounds.DEFAULT_PEAK_BYTES, (
        f"{name} allocated {peak} bytes for a {len(data)}-byte file")


@given(data=gguf_bytes())
def test_header_layout_raises_only_its_documented_types(data, tmp_path):
    path = _write(tmp_path, data)
    try:
        with open(path, "rb") as f:
            try:
                _bounds.returns_within(gguf._gguf_header_layout, f)
            except (struct.error, UnicodeDecodeError, ValueError, IndexError):
                pass
    finally:
        path.unlink(missing_ok=True)


@given(data=gguf_bytes())
def test_write_with_string_kv_raises_only_value_or_os_error(data, tmp_path):
    src = _write(tmp_path, data)
    dst = tmp_path / f"out{next(_counter)}.gguf"
    try:
        try:
            _bounds.returns_within(gguf.write_gguf_with_string_kv, src, dst, "k.new", "v")
        except (ValueError, OSError):
            pass
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
