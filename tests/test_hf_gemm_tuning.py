# SPDX-License-Identifier: AGPL-3.0-or-later
"""ROCm GEMM tuning for HF decoding (localm/inference/backends/_hf_gemm_tuning.py)."""
from __future__ import annotations

import types

import pytest

from localm.inference.backends import _hf_gemm_tuning as gt

torch = pytest.importorskip("torch")

_ROCM_GPU = bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()


class _Tunable:
    """A stand-in for torch.cuda.tunable that records calls and can fail."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def _call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name == self.fail_on:
            raise RuntimeError("boom")

    def set_filename(self, *a, **k):
        self._call("set_filename", *a, **k)

    def set_max_tuning_duration(self, *a):
        self._call("set_max_tuning_duration", *a)

    def enable(self, *a):
        self._call("enable", *a)

    def tuning_enable(self, *a):
        self._call("tuning_enable", *a)


def _fake_torch(hip, tunable):
    return types.SimpleNamespace(version=types.SimpleNamespace(hip=hip),
                                 cuda=types.SimpleNamespace(tunable=tunable), nn=torch.nn)


def test_not_rocm_does_nothing(tmp_path, monkeypatch):
    tunable = _Tunable()
    monkeypatch.setattr(gt, "decode_gemm_weights",
                        lambda torch_, model: [types.SimpleNamespace(shape=(4, 4))])
    assert gt.tune_decode_gemms(_fake_torch(None, tunable), object(), tmp_path / "r") is None
    assert tunable.calls == []
    assert not (tmp_path / "r").exists()


def test_cpu_resident_layers_are_not_tuned(tmp_path):
    tunable = _Tunable()
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))
    assert gt.decode_gemm_weights(torch, model) == []
    assert gt.tune_decode_gemms(_fake_torch("7.1", tunable), model, tmp_path / "r") is None
    assert tunable.calls == []


def test_a_tunableop_failure_turns_it_off_and_reports_nothing(tmp_path, monkeypatch):
    tunable = _Tunable(fail_on="tuning_enable")
    weight = types.SimpleNamespace(shape=(4, 4))
    monkeypatch.setattr(gt, "decode_gemm_weights", lambda torch_, model: [weight])
    monkeypatch.delenv(gt.HIPBLASLT_ENV, raising=False)
    result = gt.tune_decode_gemms(_fake_torch("7.1", tunable), object(), tmp_path / "r")
    assert result is None
    names = [c[0] for c in tunable.calls]
    assert names[-1] == "enable" and tunable.calls[-1][1] == (False,)


def test_hipblaslt_choice_set_by_the_user_is_kept(tmp_path, monkeypatch):
    tunable = _Tunable(fail_on="enable")
    monkeypatch.setattr(gt, "decode_gemm_weights",
                        lambda torch_, model: [types.SimpleNamespace(shape=(4, 4))])
    monkeypatch.setenv(gt.HIPBLASLT_ENV, "1")
    gt.tune_decode_gemms(_fake_torch("7.1", tunable), object(), tmp_path / "r")
    import os
    assert os.environ[gt.HIPBLASLT_ENV] == "1"


@pytest.mark.skipif(not _ROCM_GPU, reason="needs a ROCm GPU")
def test_rocm_tunes_each_distinct_decode_shape_once_and_leaves_tuning_off(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.Linear(512, 256), torch.nn.Linear(256, 512),
    ).to(device="cuda", dtype=torch.bfloat16)
    weights = gt.decode_gemm_weights(torch, model)
    assert [tuple(w.shape) for w in weights] == [(512, 256), (256, 512)]
    try:
        summary = gt.tune_decode_gemms(torch, model, tmp_path / "r")
        assert summary is not None and summary.startswith("tuned 2 GEMM shapes")
        assert torch.cuda.tunable.is_enabled()
        assert not torch.cuda.tunable.tuning_is_enabled()
        assert len(torch.cuda.tunable.get_results()) >= 2
        x = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ model[0].weight.float().T + model[0].bias.float()
        assert torch.allclose(model[0](x).float(), ref, atol=0.25, rtol=0.05)
    finally:
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.enable(False)
