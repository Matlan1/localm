# SPDX-License-Identifier: AGPL-3.0-or-later
"""HF backend prompt tokenization: no double-BOS.

``apply_chat_template(tokenize=False)`` already emits the model's BOS (Gemma
``<bos>``, Llama-3 ``<|begin_of_text|>``, Mistral ``<s>``). Re-tokenizing that
string with the tokenizer default (``add_special_tokens=True``) prepends a SECOND
BOS for the instruct families this backend runs, which degrades coherence. The
chat path must re-tokenize with ``add_special_tokens=False`` (matching what
``apply_chat_template(tokenize=True)`` does internally and the count_tokens path).

Tests HFWorker (``_hf_worker.py``), not the HFBackend proxy (``hf.py``):
tokenization and chat-template logic run only in the isolated child process, so
this in-process, no-subprocess unit test targets the class that owns them.
"""

from unittest.mock import MagicMock

import pytest


def test_chat_tokenization_suppresses_double_bos(monkeypatch):
    # exc_type=ImportError: transformers raises a plain ImportError (not
    # ModuleNotFoundError) when its OWN internal tokenizers version-gate fails,
    # and importorskip's default covers ModuleNotFoundError only.
    pytest.importorskip("transformers", exc_type=ImportError)
    import transformers

    from localm.inference.backends import _hf_worker as hfmod

    be = hfmod.HFWorker.__new__(hfmod.HFWorker)
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

    # chat_stream's own `from transformers import StoppingCriteriaList, ...`
    # triggers transformers' lazy-module loader to import
    # generation/stopping_criteria.py, which does a fresh `import torch`. The
    # importorskip above guards a DIFFERENT, unrelated transformers ImportError
    # and does not cover this, so the guard sits immediately before the call that
    # triggers it.
    from localm.inference.backends.llamacpp import _loader
    if _loader.native_lib_loaded():
        pytest.skip("llama.cpp's native runtime is already loaded in this "
                     "process (a real compute-device probe ran earlier in "
                     "this same pytest worker) - chat_stream's transformers "
                     "import triggers a fresh torch import, which is the "
                     "known-doomed DLL-identity conflict, not this test's "
                     "own subject")

    list(be.chat_stream([{"role": "user", "content": "hi"}], max_tokens=1))

    # The re-tokenization of the already-BOS'd template string must NOT add a
    # second BOS.
    assert tok.call_args is not None, "tokenizer was never called to encode the prompt"
    assert tok.call_args.kwargs.get("add_special_tokens") is False
