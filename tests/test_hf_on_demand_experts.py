# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kimi-Linear on transformers' own classes with the routed experts read from the checkpoint as they
are routed to: the same results as the whole model, within the byte budget. Also the kernel choice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")
modeling = pytest.importorskip("transformers.models.kimi_linear.modeling_kimi_linear")

from safetensors.torch import load_file, save_file  # noqa: E402

from localm.inference.backends import _hf_linear_attention as kernels  # noqa: E402
from localm.inference.backends import _hf_on_demand_experts as od  # noqa: E402
from localm.inference.backends import _hf_worker  # noqa: E402

EXPERTS = 6


def _config():
    return tr.AutoConfig.for_model(
        "kimi_linear", vocab_size=64, hidden_size=64, intermediate_size=64, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2, kv_lora_rank=8, qk_rope_head_dim=8,
        qk_nope_head_dim=8, v_head_dim=8, num_local_experts=EXPERTS, num_experts_per_tok=2, linear_head_dim=8,
        linear_num_heads=2, layer_types=["linear_attention", "linear_attention", "full_attention"],
        pad_token_id=0, bos_token_id=1, eos_token_id=2)


def _checkpoint(folder: Path) -> Path:
    """A checkpoint with the per-expert weight names the published one has."""
    torch.manual_seed(0)
    config = _config()
    model = tr.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.save_pretrained(folder)
    state = load_file(str(folder / "model.safetensors"))
    assert any(od._EXPERT_KEY.match(k) for k in state), sorted(state)[:5]
    return folder


@pytest.fixture
def restored_class(monkeypatch):
    monkeypatch.setattr(modeling, "KimiLinearExperts", modeling.KimiLinearExperts)


def _whole(folder):
    return tr.AutoModelForCausalLM.from_pretrained(folder, device_map="cpu", dtype=torch.float32)


def _on_demand(folder, capacity):
    od.prepare()
    model = tr.AutoModelForCausalLM.from_pretrained(folder, device_map={"": "cpu"}, dtype=torch.float32)
    store = od.attach(model, str(folder), "cpu", capacity)
    return model, store


@pytest.mark.parametrize("capacity", [0, 200_000, 10 ** 9])
def test_the_results_are_those_of_the_whole_model(tmp_path, restored_class, capacity):
    folder = _checkpoint(tmp_path / "m")
    ids = torch.randint(3, 64, (1, 80))
    whole = _whole(folder)
    with torch.no_grad():
        want = whole(input_ids=ids).logits
    want_out = whole.generate(input_ids=ids[:, :6], attention_mask=torch.ones_like(ids[:, :6]), max_new_tokens=6,
                              do_sample=False, pad_token_id=0)
    model, store = _on_demand(folder, capacity)
    assert not any("experts.gate_up_proj" in n for n, _ in model.named_parameters())
    with torch.no_grad():
        got = model(input_ids=ids).logits
    assert torch.allclose(got, want, atol=1e-4)
    assert store.held_bytes <= capacity
    out = model.generate(input_ids=ids[:, :6], attention_mask=torch.ones_like(ids[:, :6]), max_new_tokens=6,
                         do_sample=False, pad_token_id=0)
    assert torch.equal(out, want_out)


def test_prefill_then_decode_matches_a_full_forward(tmp_path, restored_class):
    folder = _checkpoint(tmp_path / "m")
    model, _ = _on_demand(folder, 10 ** 9)
    ids = torch.randint(3, 64, (1, 90))
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        out = model(input_ids=ids[:, :84], use_cache=True)
        past = out.past_key_values
        steps = [out.logits[0, -1]]
        for t in range(84, 90):
            out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            steps.append(out.logits[0, -1])
    for offset, step in enumerate(steps):
        assert torch.allclose(step, full[0, 83 + offset], atol=1e-3)


def test_only_the_experts_routed_to_are_read_and_a_second_use_reads_nothing(tmp_path, restored_class):
    folder = _checkpoint(tmp_path / "m")
    model, store = _on_demand(folder, 10 ** 9)
    with torch.no_grad():
        model(input_ids=torch.tensor([[5]]))
    first = store.reads
    # One token routes to 2 experts in each of the 2 layers with routed experts.
    assert first == 4
    with torch.no_grad():
        model(input_ids=torch.tensor([[5]]))
    assert store.reads == first and store.hits == first


def test_the_store_drops_the_least_recently_used_expert_first(tmp_path, restored_class):
    folder = _checkpoint(tmp_path / "m")
    weights = od._weight_map(folder)
    size = sum(load_file(str(folder / "model.safetensors"))[f"model.layers.1.block_sparse_moe.experts.0.w{i}.weight"].numel()
               for i in (1, 2, 3)) * 4
    store = od.ExpertStore(folder, weights, "cpu", torch.float32, size * 2)
    store.expert(1, 0)
    store.expert(1, 1)
    store.expert(1, 0)
    store.expert(1, 2)
    assert list(store._held) == [(1, 0), (1, 2)] and store.held_bytes == size * 2


def test_an_expert_is_gate_and_up_on_the_first_axis_and_down(tmp_path):
    folder = _checkpoint(tmp_path / "m")
    state = load_file(str(folder / "model.safetensors"))
    store = od.ExpertStore(folder, od._weight_map(folder), "cpu", torch.float32, 0)
    gate_up, down = store.expert(2, 3)
    prefix = "model.layers.2.block_sparse_moe.experts.3"
    assert torch.equal(gate_up, torch.cat([state[f"{prefix}.w1.weight"], state[f"{prefix}.w3.weight"]]))
    assert torch.equal(down, state[f"{prefix}.w2.weight"])


def test_it_is_used_only_for_a_kimi_linear_checkpoint_that_does_not_fit(tmp_path):
    folder = _checkpoint(tmp_path / "m")
    size = (folder / "model.safetensors").stat().st_size
    assert od.worth_using(str(folder), size - 1) is True
    assert od.worth_using(str(folder), size) is False
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config["model_type"] = "llama"
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert od.worth_using(str(folder), 0) is False
    assert od.worth_using(str(tmp_path / "absent"), 0) is False


def test_a_checkpoint_of_stacked_experts_is_not_read_this_way(tmp_path):
    folder = _checkpoint(tmp_path / "m")
    state = load_file(str(folder / "model.safetensors"))
    save_file({k: v for k, v in state.items() if "experts" not in k}, str(folder / "model.safetensors"))
    assert od.worth_using(str(folder), 0) is False


def _budget(**kwargs):
    from localm.inference import switch_admission
    return switch_admission.LoadBudget(name="m", vram_required=10 ** 11, headroom=0, resident_cap=None,
                                       pinned=frozenset(), check_split_fit=False, **kwargs)


def test_the_load_question_says_the_experts_are_read_from_disk_when_they_are():
    from localm.inference import switch_admission
    asked = switch_admission.degraded_load_confirm("m", _budget(experts_on_disk=True), 10 ** 9)
    assert asked["experts_on_disk"] is True and "reads them as they are needed" in asked["detail"]
    usual = switch_admission.degraded_load_confirm("m", _budget(), 10 ** 9)
    assert usual["experts_on_disk"] is False and "on the CPU" in usual["detail"]


def test_the_file_check_imports_neither_torch_nor_transformers():
    import ast
    source = Path(od.__file__).with_name("_hf_expert_files.py").read_text(encoding="utf-8")
    names = {n.name.split(".")[0] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import) for n in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.ImportFrom) and node.module}
    assert not names & {"torch", "transformers", "safetensors"}


class _Cuda:
    def __init__(self, free):
        self.free = free

    def mem_get_info(self, index):
        return self.free, self.free


def test_a_load_on_several_gpus_is_placed_the_usual_way(tmp_path, monkeypatch):
    folder = _checkpoint(tmp_path / "m")
    fake = type("T", (), {"cuda": _Cuda(1)})
    device = _hf_worker._on_demand_experts_device
    assert device(fake, str(folder), "cuda", {"device_map": "auto"}) == 0
    assert device(fake, str(folder), "cuda", {"device_map": "auto", "max_memory": {1: 5, "cpu": 9}}) == 1
    assert device(fake, str(folder), "cuda", {"device_map": "auto", "max_memory": {0: 5, 1: 5, "cpu": 9}}) is None
    assert device(fake, str(folder), "xpu", {"device_map": "cpu"}) is None
    assert device(type("T", (), {"cuda": _Cuda(10 ** 12)}), str(folder), "cuda", {"device_map": "auto"}) is None


FAKE_FLA = """
__version__ = "9.9.9"
"""


def _fake_fla(tmp_path, monkeypatch, kda: str):
    site = tmp_path / "site"
    pkg = site / "fla"
    (pkg / "ops" / "kda").mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_FLA, encoding="utf-8")
    (pkg / "ops" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ops" / "kda" / "__init__.py").write_text(kda, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        monkeypatch.delitem(sys.modules, name)
    import importlib
    importlib.invalidate_caches()


@pytest.fixture
def clean_fla():
    yield
    for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
        del sys.modules[name]


WORKING_KDA = """
def chunk_kda(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False):
    return v, None
fused_recurrent_kda = chunk_kda
"""
FAILING_KDA = """
def chunk_kda(*a, **k):
    raise RuntimeError('0 active drivers ([]). There should only be one.')
fused_recurrent_kda = chunk_kda
"""


def test_a_working_fla_is_used_and_named(tmp_path, monkeypatch, clean_fla):
    _fake_fla(tmp_path, monkeypatch, WORKING_KDA)
    assert kernels.choose_kernels(torch, "cpu") == "linear attention kernels: fla-core 9.9.9 (Triton)"
    assert sys.modules["fla"] is not None


def test_an_fla_that_cannot_run_is_hidden_and_the_reason_named(tmp_path, monkeypatch, clean_fla):
    _fake_fla(tmp_path, monkeypatch, FAILING_KDA)
    note = kernels.choose_kernels(torch, "cpu")
    assert note.startswith("linear attention kernels: PyTorch") and "0 active drivers" in note
    with pytest.raises(ImportError):
        import fla  # noqa: F401, PLC0415


def test_without_fla_the_pytorch_kernels_are_named(monkeypatch):
    monkeypatch.setattr(kernels.importlib.util, "find_spec", lambda name: None)
    assert kernels.choose_kernels(torch, "cpu") == "linear attention kernels: PyTorch, transformers' own"


def _reference_kda(q, k, v, g, beta):
    q, k = (x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6) for x in (q, k))
    q = q * q.shape[-1] ** -0.5
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=torch.float64)
    out = torch.zeros_like(v)
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None]
        delta = (v[:, t] - (k[:, t][..., None] * state).sum(-2)) * beta[:, t][..., None]
        state = state + k[:, t][..., None] * delta[..., None, :]
        out[:, t] = (q[:, t][..., None] * state).sum(-2)
    return out


def test_the_pytorch_kernels_agree_with_fla_at_the_models_sizes():
    """The per-token definition fla's own naive kernel computes, at the model's 32 heads of 128."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 130, 32, 128) for _ in range(3))
    g = -torch.nn.functional.softplus(torch.randn(1, 130, 32, 128)) * 2
    beta = torch.rand(1, 130, 32)
    want = _reference_kda(*(x.double() for x in (q, k, v, g, beta)))
    got, _ = modeling.chunk_kimi_delta_attention(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True,
                                                 use_qk_l2norm_in_kernel=True)
    assert (got.double() - want).abs().max() < 1e-4
