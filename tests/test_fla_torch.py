# SPDX-License-Identifier: AGPL-3.0-or-later
"""The PyTorch stand-ins for ``fla`` that let Kimi-Linear's bundled code run where the
real package cannot: Kimi Delta Attention against a plain per-token reference, the
gate, the gated norm and the short convolution against their formulas, and the
registration of the stand-in modules."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from localm.inference.backends import _fla_torch as ft  # noqa: E402
from localm.model_manager import missing_packages as mp  # noqa: E402
from localm.model_manager import model_packages as mpk  # noqa: E402

KIMI_IMPORTS = (
    "try:\n    from fla.modules import FusedRMSNormGated, ShortConvolution\n"
    "    from fla.ops.kda import chunk_kda, fused_recurrent_kda\n"
    "    from fla.ops.kda.gate import fused_kda_gate\n"
    "    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask\n"
    "    from fla.utils import tensor_cache\n"
    "except ImportError:\n    raise ImportError('Plese run `pip install -U fla-core`')\n")


def _reference(q, k, v, g, beta, state, l2norm):
    q, k, v, g, beta = (t.double() for t in (q, k, v, g, beta))
    if l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    rep = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(rep, 2) * q.shape[-1] ** -0.5
    k = k.repeat_interleave(rep, 2)
    state = state.double().clone()
    out = torch.zeros_like(v)
    for t in range(q.shape[1]):
        state = state * torch.exp(g[:, t])[..., None]
        projected = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + torch.einsum("bhk,bhv->bhkv", beta[:, t][..., None] * k[:, t],
                                     v[:, t] - projected)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], state)
    return out, state


def _inputs(batch, length, heads, value_heads, dim_k, dim_v, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen)

    q, k = rnd(batch, length, heads, dim_k), rnd(batch, length, heads, dim_k)
    v = rnd(batch, length, value_heads, dim_v)
    log_a = torch.log(torch.rand(value_heads, generator=gen) * 15 + 1)
    g = ft.fused_kda_gate(rnd(batch, length, value_heads * dim_k), log_a, dim_k,
                          g_bias=rnd(value_heads * dim_k))
    beta = torch.sigmoid(rnd(batch, length, value_heads))
    return q, k, v, g, beta


@pytest.mark.parametrize("shape", [(1, 5, 2, 2, 8, 8), (2, 64, 2, 2, 8, 12),
                                   (2, 131, 2, 4, 16, 16), (1, 200, 3, 3, 32, 16)])
@pytest.mark.parametrize("use_state", [False, True])
def test_both_forms_of_kda_match_the_per_token_reference(shape, use_state):
    batch, length, heads, value_heads, dim_k, dim_v = shape
    q, k, v, g, beta = _inputs(*shape)
    state = torch.randn(batch, value_heads, dim_k, dim_v) * 0.1 if use_state else \
        torch.zeros(batch, value_heads, dim_k, dim_v)
    want, want_state = _reference(q, k, v, g, beta, state, True)
    for kda in (ft.chunk_kda, ft.fused_recurrent_kda):
        got, got_state = kda(q, k, v, g, beta, initial_state=state if use_state else None,
                             output_final_state=True, use_qk_l2norm_in_kernel=True)
        assert got.shape == v.shape and got.dtype == v.dtype
        assert (got.double() - want).abs().max() < 1e-4
        assert (got_state.double() - want_state).abs().max() < 1e-4


def test_kda_without_the_state_or_norm_options():
    q, k, v, g, beta = _inputs(1, 70, 2, 2, 8, 8)
    out, state = ft.chunk_kda(q, k, v, g, beta)
    want, _ = _reference(q, k, v, g, beta, torch.zeros(1, 2, 8, 8), False)
    assert state is None and (out.double() - want).abs().max() < 1e-4


def test_a_split_sequence_with_the_carried_state_equals_the_whole_one():
    q, k, v, g, beta = _inputs(1, 150, 2, 2, 8, 8)
    whole, whole_state = ft.chunk_kda(q, k, v, g, beta, output_final_state=True,
                                      use_qk_l2norm_in_kernel=True)
    first, mid = ft.chunk_kda(q[:, :90], k[:, :90], v[:, :90], g[:, :90], beta[:, :90],
                              output_final_state=True, use_qk_l2norm_in_kernel=True)
    rest, last = ft.fused_recurrent_kda(q[:, 90:], k[:, 90:], v[:, 90:], g[:, 90:], beta[:, 90:],
                                        initial_state=mid, output_final_state=True,
                                        use_qk_l2norm_in_kernel=True)
    assert (torch.cat([first, rest], 1) - whole).abs().max() < 1e-4
    assert (last - whole_state).abs().max() < 1e-4


def test_variable_length_sequences_run_independently():
    q, k, v, g, beta = _inputs(1, 100, 2, 2, 8, 8)
    cu = torch.tensor([0, 37, 38, 100])
    state0 = torch.randn(3, 2, 8, 8) * 0.1
    for kda in (ft.chunk_kda, ft.fused_recurrent_kda):
        out, state = kda(q, k, v, g, beta, initial_state=state0, output_final_state=True,
                         use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        for i, (a, b) in enumerate([(0, 37), (37, 38), (38, 100)]):
            want, want_state = _reference(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b],
                                          beta[:, a:b], state0[i:i + 1], True)
            assert (out[:, a:b].double() - want).abs().max() < 1e-4
            assert (state[i:i + 1].double() - want_state).abs().max() < 1e-4


def test_low_precision_input_keeps_its_dtype_and_stays_finite():
    q, k, v, g, beta = _inputs(1, 130, 2, 2, 8, 8)
    out, _ = ft.chunk_kda(q.bfloat16(), k.bfloat16(), v.bfloat16(), g, beta,
                          use_qk_l2norm_in_kernel=True)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


def test_the_gate_is_minus_exp_a_times_softplus():
    raw = torch.randn(2, 5, 3 * 8)
    log_a = torch.randn(1, 1, 3, 1)
    bias = torch.randn(24)
    want = -log_a.reshape(3, 1).exp() * torch.nn.functional.softplus((raw + bias).view(2, 5, 3, 8))
    got = ft.fused_kda_gate(raw, log_a, 8, g_bias=bias)
    assert got.shape == (2, 5, 3, 8) and got.dtype == torch.float32
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("activation,gate", [("sigmoid", torch.sigmoid),
                                             ("swish", lambda t: t * torch.sigmoid(t))])
def test_the_gated_norm_is_rms_norm_times_the_activated_gate(activation, gate):
    norm = ft.FusedRMSNormGated(16, eps=1e-5, activation=activation)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
    x, g = torch.randn(3, 4, 2, 16), torch.randn(3, 4, 2, 16)
    want = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight * gate(g)
    assert torch.allclose(norm(x, g), want, atol=1e-5)
    assert list(norm.state_dict()) == ["weight"]


def test_the_short_convolution_is_causal_and_carries_its_state():
    conv = ft.ShortConvolution(12, 4, activation="silu")
    torch.nn.init.normal_(conv.weight)
    assert list(conv.state_dict()) == ["weight"] and tuple(conv.weight.shape) == (12, 1, 4)
    x = torch.randn(2, 20, 12)
    padded = torch.nn.functional.pad(x.transpose(1, 2), (3, 0))
    want = torch.nn.functional.silu(
        torch.nn.functional.conv1d(padded, conv.weight, groups=12)).transpose(1, 2)
    whole, whole_state = conv(x, output_final_state=True)
    assert torch.allclose(whole, want, atol=1e-6)
    head, state = conv(x[:, :13], output_final_state=True)
    tail, state = conv(x[:, 13:], cache=state, output_final_state=True)
    assert torch.allclose(torch.cat([head, tail], 1), whole, atol=1e-6)
    assert torch.allclose(state, whole_state)
    state, steps = None, []
    for t in range(20):
        step, state = conv(x[:, t:t + 1], cache=state, output_final_state=True)
        steps.append(step)
    assert torch.allclose(torch.cat(steps, 1), whole, atol=1e-6)
    assert conv(x)[1] is None


def test_the_short_convolution_restarts_at_each_variable_length_sequence():
    conv = ft.ShortConvolution(6, 4)
    torch.nn.init.normal_(conv.weight)
    x = torch.randn(1, 30, 6)
    cu = torch.tensor([0, 11, 30])
    got, state = conv(x, output_final_state=True, cu_seqlens=cu)
    first, first_state = conv(x[:, :11], output_final_state=True)
    second, second_state = conv(x[:, 11:], output_final_state=True)
    assert torch.allclose(got, torch.cat([first, second], 1), atol=1e-6)
    assert torch.allclose(state, torch.cat([first_state, second_state], 0))


def test_the_mask_helpers_count_the_unpadded_tokens():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert ft.prepare_lens_from_mask(mask).tolist() == [2, 3]
    assert ft.prepare_cu_seqlens_from_mask(mask).tolist() == [0, 2, 5]


@pytest.fixture
def no_fla(monkeypatch):
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(ft, "_real_fla_imports", lambda: False)


def _kimi_folder(tmp_path: Path, source: str = KIMI_IMPORTS) -> Path:
    folder = tmp_path / "kimi"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(
        {"auto_map": {"AutoModelForCausalLM": "modeling_x.X"}}), encoding="utf-8")
    (folder / "modeling_x.py").write_text(source, encoding="utf-8")
    return folder


def test_provide_for_registers_the_names_the_model_imports(tmp_path, no_fla):
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is True
    probe = tmp_path / "kimi_probe.py"
    probe.write_text(KIMI_IMPORTS, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("kimi_probe", probe)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.chunk_kda is ft.chunk_kda
    assert module.ShortConvolution is ft.ShortConvolution
    assert getattr(sys.modules["fla"], ft.STAND_IN_MARK)


def test_provide_for_leaves_a_model_that_does_not_import_fla_alone(tmp_path, no_fla):
    folder = _kimi_folder(tmp_path, "import torch\n")
    assert ft.provide_for(str(folder)) is False
    assert "fla" not in sys.modules


def test_provide_for_prefers_a_working_fla_install(tmp_path, no_fla, monkeypatch):
    monkeypatch.setattr(ft, "_real_fla_imports", lambda: True)
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is False
    assert "fla" not in sys.modules


def test_provide_for_replaces_an_fla_that_is_installed_but_does_not_import(tmp_path, monkeypatch):
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        monkeypatch.delitem(sys.modules, name)
    pkg = tmp_path / "site" / "fla"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "modules.py").write_text("import localm_no_such_dependency_zz\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    assert ft._real_fla_imports() is False
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is True
    assert getattr(sys.modules["fla"], ft.STAND_IN_MARK)


def test_a_model_needing_only_fla_is_neither_missing_nor_refused(tmp_path, monkeypatch):
    from localm.inference.backends import hf as hf_backend
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: True)
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "fla")
    monkeypatch.setattr(sys, "platform", "win32")
    folder = _kimi_folder(tmp_path)
    assert "fla" in mp.folder_third_party_imports(folder)
    assert mp.missing_modules(folder) == []
    assert mp.folder_refusal(folder) is None
    state = mpk.folder_state(folder)
    assert state.missing == [] and state.addable == []
