# SPDX-License-Identifier: AGPL-3.0-or-later
"""A chat request to an HF worker that holds no model is refused as not loaded,
not as a missing chat template."""
import pytest

from localm.inference.backends._hf_worker import HFWorker


def _chat(worker):
    return next(worker.chat_stream([{"role": "user", "content": "hi"}]))


def test_a_worker_that_never_loaded_says_so(tmp_path):
    worker = HFWorker(str(tmp_path))
    with pytest.raises(RuntimeError, match="Model not loaded"):
        _chat(worker)


def test_a_worker_that_was_unloaded_says_so(tmp_path):
    worker = HFWorker(str(tmp_path))
    worker.unload()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        _chat(worker)
