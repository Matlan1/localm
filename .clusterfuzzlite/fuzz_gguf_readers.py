# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coverage-guided fuzz target for the GGUF header readers.

Every reader must answer "no signal" for a file it cannot parse; any exception
escaping one is a crash."""
import os
import pathlib
import sys
import tempfile

import atheris

os.environ.setdefault("LOCALM_HOME", tempfile.mkdtemp(prefix="localm-fuzz-"))

with atheris.instrument_imports():
    from localm.model_manager import gguf

READERS = (
    gguf.gguf_kv_bytes_per_token,
    gguf.gguf_nextn_predict_layers,
    gguf.gguf_expert_count,
    gguf.gguf_expert_counts,
    gguf.gguf_block_bytes,
    gguf.gguf_moe_expert_bytes_by_layer,
    gguf.gguf_input_layer_bytes,
    gguf.gguf_recurrent_state_bytes,
    gguf.gguf_split_layout,
    gguf._gguf_declared_min_size,
    gguf._gguf_metadata_probe,
    gguf.gguf_pretokenizer,
    gguf.gguf_embedding_signal,
    gguf.gguf_architecture,
    gguf.gguf_is_mmproj,
    gguf.gguf_mmproj_inferred_projector_type,
    gguf._has_gguf_magic,
)

_PATH = os.path.join(tempfile.mkdtemp(prefix="localm-fuzz-gguf-"), "input.gguf")


def test_one_input(data: bytes) -> None:
    with open(_PATH, "wb") as f:
        f.write(data)
    for reader in READERS:
        reader(pathlib.Path(_PATH))


if __name__ == "__main__":
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
