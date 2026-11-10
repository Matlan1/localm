# SPDX-License-Identifier: AGPL-3.0-or-later
"""A CPU load of a model larger than the available RAM keeps the rest on disk, reading the
model's own files, instead of running out of memory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

psutil = pytest.importorskip("psutil")

from localm.inference.backends import _hf_worker  # noqa: E402


def _available(monkeypatch, nbytes: int) -> None:
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=nbytes))


def _weights(folder, *sizes) -> None:
    for i, size in enumerate(sizes):
        with open(folder / f"model-{i}.safetensors", "wb") as f:
            f.truncate(size)


def test_a_model_that_fits_goes_to_ram_at_the_cpu_dtype(tmp_path, monkeypatch):
    _weights(tmp_path, 100, 200)
    _available(monkeypatch, 1000)
    assert _hf_worker._cpu_device_map(str(tmp_path)) == ({"device_map": "cpu"}, False)


def test_a_model_larger_than_ram_at_the_cpu_dtype_gets_a_budget_and_keeps_its_dtype(tmp_path, monkeypatch):
    _weights(tmp_path, 300, 300)
    _available(monkeypatch, 1000)
    kwargs, keep = _hf_worker._cpu_device_map(str(tmp_path))
    assert kwargs == {"device_map": "auto", "max_memory": {"cpu": 800}} and keep is True


def test_the_limit_is_ninety_percent_of_the_available_ram(tmp_path, monkeypatch):
    _weights(tmp_path, 450)
    _available(monkeypatch, 1000)
    assert _hf_worker._cpu_device_map(str(tmp_path))[1] is False
    _weights(tmp_path, 451)
    assert _hf_worker._cpu_device_map(str(tmp_path))[1] is True


def test_a_folder_without_safetensors_goes_to_ram(tmp_path, monkeypatch):
    _available(monkeypatch, 1000)
    assert _hf_worker._cpu_device_map(str(tmp_path))[0] == {"device_map": "cpu"}
    assert _hf_worker._cpu_device_map(str(tmp_path / "absent"))[0] == {"device_map": "cpu"}


def test_the_overflow_placement_reads_the_rest_from_the_models_own_files(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    config = tr.LlamaConfig(hidden_size=32, num_attention_heads=2, num_hidden_layers=4,
                            intermediate_size=32, vocab_size=64)
    torch.manual_seed(0)
    tr.LlamaForCausalLM(config).save_pretrained(tmp_path)
    files_before = sorted(p.name for p in tmp_path.iterdir())
    ids = torch.tensor([[1, 2, 3, 4]])
    whole = tr.LlamaForCausalLM.from_pretrained(tmp_path, device_map="cpu")
    with torch.no_grad():
        want = whole(ids).logits
    file_bytes = _hf_worker._safetensors_bytes(str(tmp_path))
    _available(monkeypatch, file_bytes // 2)
    kwargs, keep = _hf_worker._cpu_device_map(str(tmp_path))
    assert keep is True
    model = tr.LlamaForCausalLM.from_pretrained(tmp_path, dtype="auto", offload_buffers=True, **kwargs)
    assert "disk" in model.hf_device_map.values()
    with torch.no_grad():
        got = model(ids).logits
    assert torch.allclose(got, want, atol=1e-5)
    assert sorted(p.name for p in tmp_path.iterdir()) == files_before
