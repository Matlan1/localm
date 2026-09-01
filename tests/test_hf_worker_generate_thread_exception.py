# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression test: HFWorker.chat_stream() must not hang forever when
model.generate() raises inside its own background thread.

model.generate() runs on a daemon thread while chat_stream() streams tokens
from the TextIteratorStreamer that thread feeds. threading.Thread never
propagates an exception from its target to any other thread, so without an
explicit capture the streamer's queue never receives a stop signal and the
`for token_text in streamer` loop blocks forever. chat_stream() must instead
re-raise the same exception on its own caller once the thread has finished.

Tests HFWorker (_hf_worker.py) directly, in-process, no subprocess and no
real model - same rationale and mocking shape as
test_hf_prompt_tokenization.py.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from localm.inference.backends._hf_worker import HFWorker

_MESSAGES = [{"role": "user", "content": "hi"}]


def _make_worker(model, tokenizer) -> HFWorker:
    worker = HFWorker.__new__(HFWorker)
    worker._processor = None
    worker._is_multimodal = False
    worker.context_capacity = None
    worker._model = model
    worker._tokenizer = tokenizer
    worker.last_finish_reason = "stop"
    return worker


def _fake_model_and_tokenizer():
    model = MagicMock()
    model.device = "cpu"
    model.generation_config = None
    tokenizer = MagicMock()
    tokenizer.eos_token_id = None
    tokenizer.apply_chat_template.return_value = "fake prompt"
    tokenizer.return_value.to.return_value = {
        "input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]],
    }
    return model, tokenizer


def _drain_with_timeout(gen, timeout: float):
    """Consume *gen* on its own daemon thread, bounded by *timeout* seconds.

    A hung generator leaves that thread running in the background (harmless -
    it is a daemon) rather than blocking this call forever. Returns
    (finished, chunks, errors)."""
    chunks: list = []
    errors: list = []
    finished = threading.Event()

    def _drain():
        try:
            for chunk in gen:
                chunks.append(chunk)
        except BaseException as exc:  # noqa: BLE001 - captured, not swallowed
            errors.append(exc)
        finally:
            finished.set()

    threading.Thread(target=_drain, daemon=True).start()
    finished.wait(timeout)
    return finished.is_set(), chunks, errors


@pytest.fixture(autouse=True)
def _skip_if_native_runtime_already_loaded():
    # chat_stream's own `from transformers import StoppingCriteriaList, ...`
    # triggers a fresh `import torch`, the known-doomed DLL-identity conflict
    # with llama.cpp's native runtime if that already loaded earlier in this
    # same pytest worker - see test_hf_prompt_tokenization.py.
    from localm.inference.backends.llamacpp import _loader
    if _loader.native_lib_loaded():
        pytest.skip("llama.cpp's native runtime is already loaded in this "
                     "process; a fresh torch import here is the known-doomed "
                     "DLL-identity conflict, not this test's own subject")


class TestGenerateThreadExceptionSurfacesPromptly:
    def test_generate_exception_surfaces_instead_of_hanging(self):
        pytest.importorskip("transformers", exc_type=ImportError)
        model, tokenizer = _fake_model_and_tokenizer()
        model.generate.side_effect = ValueError("boom from generate")
        worker = _make_worker(model, tokenizer)

        gen = worker.chat_stream(_MESSAGES)
        finished, chunks, errors = _drain_with_timeout(gen, timeout=5.0)

        assert finished, (
            "chat_stream() did not surface the generate() exception within "
            "5s - it is hanging instead of propagating the failure"
        )
        # TextStreamer.end() queues one "" marker ahead of its stop signal
        # even with an empty token_cache - already true of a real generate()
        # call's own normal completion, not something this fix introduces.
        # The property that matters here is that no REAL content leaks out.
        assert all(c == "" for c in chunks), (
            f"expected no real generated content before the exception, got {chunks!r}"
        )
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert "boom from generate" in str(errors[0])

    def test_normal_generation_completes_with_no_error(self):
        pytest.importorskip("transformers", exc_type=ImportError)
        model, tokenizer = _fake_model_and_tokenizer()
        # generate() returns normally without ever touching the streamer -
        # the closest a bare Mock gets to a real call.
        worker = _make_worker(model, tokenizer)

        gen = worker.chat_stream(_MESSAGES)
        finished, chunks, errors = _drain_with_timeout(gen, timeout=5.0)

        assert finished, "chat_stream() hung on a generate() call that raised nothing"
        assert errors == []
