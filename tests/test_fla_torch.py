# SPDX-License-Identifier: AGPL-3.0-or-later
"""The PyTorch stand-ins for ``fla`` that let Kimi-Linear's bundled code run where the
real package cannot: Kimi Delta Attention against a plain per-token reference, the
gate, the gated norm and the short convolution against their formulas, and the
registration of the stand-in modules."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
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
                                   (2, 131, 2, 4, 16, 16), (1, 200, 3, 3, 32, 16),
                                   (1, 130, 32, 32, 128, 128)])
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


@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_the_gate_and_beta_can_be_computed_in_the_kernel(lower_bound):
    q, k, v, _, _ = _inputs(1, 70, 2, 2, 8, 8)
    raw, bias, log_a = torch.randn(1, 70, 2, 8), torch.randn(16), torch.randn(2)
    beta_raw = torch.randn(1, 70, 2)
    if lower_bound is None:
        g = -log_a.view(2, 1).exp() * torch.nn.functional.softplus(raw + bias.view(2, 8))
    else:
        g = lower_bound * torch.sigmoid(log_a.view(2, 1).exp() * (raw + bias.view(2, 8)))
    want, want_state = _reference(q, k, v, g, beta_raw.sigmoid(), torch.zeros(1, 2, 8, 8), True)
    for kda, extra in ((ft.chunk_kda, {"safe_gate": lower_bound is not None}), (ft.fused_recurrent_kda, {})):
        got, state = kda(q, k, v, raw, beta_raw, A_log=log_a, dt_bias=bias, output_final_state=True,
                         use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                         use_beta_sigmoid_in_kernel=True, lower_bound=lower_bound,
                         transpose_state_layout=True, **extra)
        assert (got.double() - want).abs().max() < 1e-4
        assert (state.double() - want_state).abs().max() < 1e-4


@pytest.mark.parametrize("kda", [ft.chunk_kda, ft.fused_recurrent_kda])
def test_an_option_the_kernels_do_not_have_raises_instead_of_changing_the_result(kda):
    q, k, v, g, beta = _inputs(1, 5, 2, 2, 8, 8)
    with pytest.raises(TypeError, match="allow_neg_eigval"):
        kda(q, k, v, g, beta, allow_neg_eigval=True)


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


def _convolved(conv, x, state):
    width = conv.kernel_size[0]
    xin = torch.cat([state[..., -(width - 1):], x.transpose(1, 2)], dim=-1)
    return torch.nn.functional.silu(
        torch.nn.functional.conv1d(xin, conv.weight, groups=conv.hidden_size)).transpose(1, 2)


def test_the_short_convolution_restarts_at_each_variable_length_sequence():
    conv = ft.ShortConvolution(6, 4)
    torch.nn.init.normal_(conv.weight)
    x = torch.randn(1, 30, 6)
    cu = torch.tensor([0, 11, 11, 30])
    got, state = conv(x, output_final_state=True, cu_seqlens=cu)
    zero = torch.zeros(1, 6, 4)
    first, second = _convolved(conv, x[:, :11], zero), _convolved(conv, x[:, 11:], zero)
    assert torch.allclose(got, torch.cat([first, second], 1), atol=1e-6)
    assert state.shape == (3, 6, 4)
    assert torch.allclose(state[0], x[:, 7:11].transpose(1, 2)[0])
    assert torch.equal(state[1], torch.zeros(6, 4))
    assert torch.allclose(state[2], x[:, 26:30].transpose(1, 2)[0])


def test_each_variable_length_sequence_continues_from_its_own_state():
    conv = ft.ShortConvolution(6, 4)
    torch.nn.init.normal_(conv.weight)
    x = torch.randn(1, 30, 6)
    cu = torch.tensor([0, 11, 11, 30])
    cache = torch.randn(3, 6, 4)
    got, state = conv(x, cache=cache, output_final_state=True, cu_seqlens=cu)
    first = _convolved(conv, x[:, :11], cache[0:1])
    second = _convolved(conv, x[:, 11:], cache[2:3])
    assert torch.allclose(got, torch.cat([first, second], 1), atol=1e-6)
    assert torch.equal(state[1], cache[1])
    assert torch.allclose(state[2], x[:, 26:30].transpose(1, 2)[0])


def test_the_mask_helpers_count_the_unpadded_tokens():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert ft.prepare_lens_from_mask(mask).tolist() == [2, 3]
    assert ft.prepare_cu_seqlens_from_mask(mask).tolist() == [0, 2, 5]


@pytest.fixture
def no_fla(monkeypatch):
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(ft, "_real_fla_works", lambda _path, _device: False)
    monkeypatch.setattr(ft, "_real_failure", None)


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


FAKE_RELEASE = """
__version__ = "9.9.9"
"""

FAKE_OLD_GATE = """
from localm.inference.backends import _fla_torch as ft
def fused_kda_gate(g, A, head_k_dim, g_bias=None, beta=1.0, threshold=20.0):
    return ft.fused_kda_gate(g, A, head_k_dim, g_bias=g_bias)
"""

# The signature of the current releases: g is [..., H, K], A_log has H elements.
FAKE_NEW_GATE = """
import torch
def fused_kda_gate(g, A_log=None, dt_bias=None, lower_bound=None, output_dtype=torch.float32):
    assert g.shape[-1] != A_log.numel() * g.shape[-2] and A_log.dim() == 1
    assert dt_bias.dim() == 1 and g.dtype == torch.float32
    H, K = g.shape[-2:]
    assert A_log.numel() == H and dt_bias.numel() == H * K
    return -A_log.view(H, 1).exp() * torch.nn.functional.softplus(g + dt_bias.view(H, K))
"""

FAKE_KERNELS = """
from localm.inference.backends import _fla_torch as ft
chunk_kda = ft.chunk_kda
fused_recurrent_kda = ft.fused_recurrent_kda
"""

FAKE_MODULES = """
from localm.inference.backends import _fla_torch as ft
class FusedRMSNormGated(ft.FusedRMSNormGated):
    pass
class ShortConvolution(ft.ShortConvolution):
    pass
"""


def _fake_fla(tmp_path, monkeypatch, gate=FAKE_NEW_GATE, kernels=FAKE_KERNELS, modules=FAKE_MODULES):
    site = tmp_path / "site"
    pkg = site / "fla"
    for sub in (pkg, pkg / "ops", pkg / "ops" / "kda", pkg / "ops" / "utils"):
        sub.mkdir(parents=True)
        (sub / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text(FAKE_RELEASE, encoding="utf-8")
    (pkg / "modules.py").write_text(modules, encoding="utf-8")
    (pkg / "ops" / "kda" / "__init__.py").write_text(kernels, encoding="utf-8")
    (pkg / "ops" / "kda" / "gate.py").write_text(gate, encoding="utf-8")
    (pkg / "ops" / "utils" / "index.py").write_text(
        "from localm.inference.backends._fla_torch import prepare_lens_from_mask, "
        "prepare_cu_seqlens_from_mask\n", encoding="utf-8")
    (pkg / "utils.py").write_text("def tensor_cache(fn):\n    return fn\n", encoding="utf-8")
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(ft, "_real_failure", None)
    importlib.invalidate_caches()


@pytest.fixture
def clean_fla():
    yield
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        del sys.modules[name]


def test_a_working_fla_install_is_chosen_and_says_so(tmp_path, monkeypatch, clean_fla):
    _fake_fla(tmp_path, monkeypatch)
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is False
    assert not getattr(sys.modules["fla"], ft.STAND_IN_MARK, False)
    assert sys.modules["fla.ops.kda"].chunk_kda is ft.chunk_kda
    assert ft.kernel_note() == "linear attention kernels: fla-core 9.9.9 (Triton)"


def test_the_gate_of_a_current_release_takes_the_call_the_model_makes(tmp_path, monkeypatch, clean_fla):
    _fake_fla(tmp_path, monkeypatch)
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is False
    raw, bias = torch.randn(1, 5, 3 * 8).bfloat16(), torch.randn(24)
    log_a = torch.randn(1, 1, 3, 1)
    got = sys.modules["fla.ops.kda.gate"].fused_kda_gate(raw, log_a, 8, g_bias=bias)
    assert torch.allclose(got, ft.fused_kda_gate(raw, log_a, 8, g_bias=bias), atol=1e-6)


def test_a_release_with_the_gate_the_model_was_written_for_is_not_wrapped(tmp_path, monkeypatch, clean_fla):
    _fake_fla(tmp_path, monkeypatch, gate=FAKE_OLD_GATE)
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is False
    gate = sys.modules["fla.ops.kda.gate"].fused_kda_gate
    assert "g_bias" in inspect.signature(gate).parameters and "beta" in inspect.signature(gate).parameters


@pytest.mark.parametrize("kernels,reason", ids=["raises", "differs", "nan"], argvalues=[
    ("def chunk_kda(*a, **k):\n    raise RuntimeError('0 active drivers ([]). There should only be one.')\n"
     "fused_recurrent_kda = chunk_kda\n", "0 active drivers"),
    (FAKE_KERNELS + "_good = chunk_kda\ndef chunk_kda(q, k, v, g, beta, **kw):\n"
     "    o, s = _good(q, k, v, g, beta, **kw)\n    return o + 0.5, s\n", "chunked KDA differs"),
    (FAKE_KERNELS + "_good = fused_recurrent_kda\ndef fused_recurrent_kda(q, k, v, g, beta, **kw):\n"
     "    o, s = _good(q, k, v, g, beta, **kw)\n    return o, s * float('nan')\n", "recurrent KDA state differs"),
])
def test_an_installed_fla_whose_kernels_fail_or_differ_gives_way(tmp_path, monkeypatch, clean_fla, kernels, reason):
    _fake_fla(tmp_path, monkeypatch, kernels=kernels)
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is True
    assert getattr(sys.modules["fla"], ft.STAND_IN_MARK)
    assert sys.modules["fla.ops.kda"].chunk_kda is ft.chunk_kda
    note = ft.kernel_note()
    assert note.startswith("linear attention kernels: localm's PyTorch kernels; the installed fla-core")
    assert reason in note


def test_an_installed_fla_whose_gate_cannot_take_the_call_gives_way(tmp_path, monkeypatch, clean_fla):
    _fake_fla(tmp_path, monkeypatch, gate="def fused_kda_gate(g):\n    return g\n")
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is True
    assert "TypeError" in ft.kernel_note()


def test_provide_for_replaces_an_fla_that_is_installed_but_does_not_import(tmp_path, monkeypatch, clean_fla):
    _fake_fla(tmp_path, monkeypatch, modules="import localm_no_such_dependency_zz\n")
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is True
    assert getattr(sys.modules["fla"], ft.STAND_IN_MARK)
    assert "localm_no_such_dependency_zz" in ft.kernel_note()


def test_the_check_runs_at_the_sizes_in_the_models_config(tmp_path, monkeypatch, clean_fla):
    folder = _kimi_folder(tmp_path)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config["linear_attn_config"] = {"num_heads": 3, "head_dim": 16, "short_conv_kernel_size": 5}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert ft._kernel_sizes(str(folder)) == (3, 16, 5)
    seen = []
    monkeypatch.setattr(ft, "_check_real_kernels", lambda *a: seen.append(a))
    _fake_fla(tmp_path, monkeypatch)
    assert ft.provide_for(str(folder), "cpu") is False
    assert seen == [("cpu", 3, 16, 5)]


def test_a_config_without_the_sizes_gets_small_ones(tmp_path):
    assert ft._kernel_sizes(str(_kimi_folder(tmp_path))) == (2, 8, 4)
    assert ft._kernel_sizes(str(tmp_path / "absent")) == (2, 8, 4)


def test_the_note_offers_the_package_only_where_localm_can_add_it(tmp_path, no_fla, monkeypatch, clean_fla):
    assert ft.kernel_note() is None
    assert ft.provide_for(str(_kimi_folder(tmp_path))) is True
    monkeypatch.setattr(mpk, "_importable", lambda m: m == "triton")
    monkeypatch.setattr(sys, "platform", "linux")
    assert ft.kernel_note() == ("linear attention kernels: localm's PyTorch kernels; "
                                "`localm install-packages` adds the Triton kernels of fla-core")
    monkeypatch.setattr(mpk, "_importable", lambda m: False)
    assert ft.kernel_note() == "linear attention kernels: localm's PyTorch kernels"
    monkeypatch.setattr(mpk, "_importable", lambda m: True)
    monkeypatch.setattr(sys, "platform", "win32")
    assert ft.kernel_note() == "linear attention kernels: localm's PyTorch kernels"


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


def test_linux_with_triton_loads_without_fla_and_offers_it(tmp_path, monkeypatch):
    from localm.inference.backends import hf as hf_backend
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: True)
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "fla")
    monkeypatch.setattr(mpk, "_importable", lambda m: m == "triton")
    monkeypatch.setattr(sys, "platform", "linux")
    folder = _kimi_folder(tmp_path)
    assert mp.folder_refusal(folder) is None
    state = mpk.folder_state(folder)
    assert state.missing == [] and state.optional == ["fla"] and state.to_add == ["fla"]
    assert state.as_dict() == {"missing": [], "addable": ["fla-core"], "installable": True}


def test_linux_without_triton_loads_without_fla_and_offers_nothing(tmp_path, monkeypatch):
    from localm.inference.backends import hf as hf_backend
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: True)
    monkeypatch.setattr(mp, "is_importable", lambda m: m != "fla")
    monkeypatch.setattr(mpk, "_importable", lambda m: False)
    monkeypatch.setattr(sys, "platform", "linux")
    state = mpk.folder_state(_kimi_folder(tmp_path))
    assert state.missing == [] and state.to_add == []


def test_an_install_that_has_fla_offers_nothing(tmp_path, monkeypatch):
    from localm.inference.backends import hf as hf_backend
    monkeypatch.setattr(hf_backend, "_trust_remote_code_enabled", lambda: True)
    monkeypatch.setattr(mp, "is_importable", lambda m: True)
    monkeypatch.setattr(mpk, "_importable", lambda m: True)
    monkeypatch.setattr(sys, "platform", "linux")
    state = mpk.folder_state(_kimi_folder(tmp_path))
    assert state.missing == [] and state.to_add == []
