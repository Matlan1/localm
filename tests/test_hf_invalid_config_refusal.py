# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Hugging Face model folder whose config.json holds a value transformers'
strict validation rejects (a 4300-digit hidden size) is refused with one
sentence that names the bad value shortened, not an unexpected-error traceback."""

from __future__ import annotations

import json
import re

import pytest

from localm.inference.backends._hf_runner import HFRunner
from localm.inference.backends._hf_worker import HFWorker
from localm.inference.backends.base import ModelRefusedError

pytest.importorskip("transformers")

HUGE = "9" * 4300


@pytest.fixture()
def bad_config_folder(tmp_path):
    folder = tmp_path / "badcfg"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": int(HUGE), "num_hidden_layers": 2,
        "num_attention_heads": 32, "vocab_size": 100}), encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"\0" * 64)
    return folder


def _assert_clean(message: str) -> None:
    assert "hidden size" in message
    assert "4300 digits" in message
    assert not re.search(r"\d{40,}", message)
    assert len(message) < 600
    assert "\n" not in message


def test_the_worker_refuses_a_config_with_a_huge_value(bad_config_folder):
    with pytest.raises(ModelRefusedError) as caught:
        HFWorker(str(bad_config_folder), device="cpu").load()
    _assert_clean(str(caught.value))


def test_the_runner_hands_the_parent_a_refusal_not_a_fault(bad_config_folder):
    runner = HFRunner()
    try:
        with pytest.raises(ModelRefusedError) as caught:
            runner.spawn_and_load({"model_path": str(bad_config_folder), "device": "cpu"},
                                  timeout=300)
    finally:
        runner.shutdown(grace=0)
    _assert_clean(str(caught.value))
