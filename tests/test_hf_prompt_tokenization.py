# SPDX-License-Identifier: AGPL-3.0-or-later
"""HF backend prompt tokenization: no double-BOS (U-1).

``apply_chat_template(tokenize=False)`` already emits the model's BOS (Gemma
``<bos>``, Llama-3 ``<|begin_of_text|>``, Mistral ``<s>``). Re-tokenizing that
string with the tokenizer default (``add_special_tokens=True``) prepends a SECOND
BOS for the instruct families this backend runs, which degrades coherence. The
chat path must re-tokenize with ``add_special_tokens=False`` (matching what
``apply_chat_template(tokenize=True)`` does internally and the count_tokens path).
"""

from unittest.mock import MagicMock

import pytest


def test_chat_tokenization_suppresses_double_bos(monkeypatch):
    # exc_type=ImportError: transformers raises a plain ImportError (not
    # ModuleNotFoundError) when its OWN internal tokenizers version-gate fails
    # (a mismatched transitive pin, not a "not installed" case) - pytest 9.1
    # narrowed importorskip's default to ModuleNotFoundError only, so without
    # this the test hard-fails instead of skipping on exactly the case this
    # skip exists for.
    pytest.importorskip("transformers", exc_type=ImportError)
    import transformers

    from localm.inference.backends import hf as hfmod

    be = hfmod.HFBackend.__new__(hfmod.HFBackend)
    be._processor = None
    be._is_multimodal = False
    be._model = MagicMock()

    tok = MagicMock()
    tok.apply_chat_template.return_value = "<bos>hello"
    # tokenizer(text, ...).to(device) -> a real mapping so **inputs spreads.
    tok.return_value.to.return_value = {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
    be._tokenizer = tok

    class _FakeStreamer:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            return iter(())   # yield nothing -> the generate thread is a no-op

    monkeypatch.setattr(transformers, "TextIteratorStreamer", _FakeStreamer)
    monkeypatch.setattr(hfmod, "_grammar_processor", lambda *a, **k: None)

    list(be.chat_stream([{"role": "user", "content": "hi"}], max_tokens=1))

    # The re-tokenization of the already-BOS'd template string must NOT add a
    # second BOS.
    assert tok.call_args is not None, "tokenizer was never called to encode the prompt"
    assert tok.call_args.kwargs.get("add_special_tokens") is False
