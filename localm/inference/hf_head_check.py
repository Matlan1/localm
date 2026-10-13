# SPDX-License-Identifier: AGPL-3.0-or-later
"""Whether a Hugging Face sequence-classification checkpoint carries its head.

A cross-encoder reranker is a transformer plus a classification head. A conversion
that dropped the head still loads, with the head initialised at random, and then
returns arbitrary numbers as relevance scores. ``classifier_head_present`` reads
the tensor names from the safetensors headers, without loading any weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from localm.debuglog import logger

# Tensor-name prefixes of a sequence-classification head: ``classifier.*`` for the
# BERT, RoBERTa, XLM-R, DeBERTa and ModernBERT families, ``score.*`` for the
# decoder-based ones (Qwen2, Llama, Gemma), ``classification_head.*`` for Bart,
# MBart and T5, ``logits_proj.*`` for XLNet and XLM.
HEAD_PREFIXES = ("classifier.", "score.", "classification_head.", "logits_proj.")


def classifier_head_present(files: list) -> Optional[bool]:
    """True when a tensor named like a classification head is in *files* (the
    safetensors shards of a model), False when every file was read and none is,
    and None when there are no files or one cannot be read."""
    if not files:
        return None
    from localm.inference.backends._hf_fp8 import _safetensors_header
    for path in files:
        try:
            names = _safetensors_header(Path(path)).keys()
        except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
            logger.debug("hf head check: could not read %s: %s", path, e)
            return None
        if any(name.startswith(HEAD_PREFIXES) for name in names):
            return True
    return False
