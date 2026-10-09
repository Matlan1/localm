# SPDX-License-Identifier: AGPL-3.0-or-later
"""FP8 checkpoints in the HF backend and the worker's Hub kernel gate.

The gate tests run in a fresh interpreter (the gate edits ``sys.modules`` for
the whole process). The real-transformers tests build a tiny checkpoint and
load it through ``HFWorker`` on CPU, recording socket and import audit events
from the start of that interpreter.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import localm
from localm.inference.backends import _hf_fp8
from localm.inference.backends._hf_worker import (
    _dequantized_note,
    _hub_kernel_blocked_message,
    _refuse_fp8_too_big,
)

_REPO_ROOT = str(Path(localm.__file__).resolve().parent.parent)

FP8_QC = {"quant_method": "fp8", "activation_scheme": "dynamic",
          "weight_block_size": [128, 128]}
KERNEL_SHA = "0123456789abcdef0123456789abcdef01234567"


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Write a safetensors file whose header lists *tensors* ({name: (dtype,
    shape)}) with zero-filled data of the right length."""
    itemsize = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "I64": 8, "BOOL": 1}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = itemsize[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def _model_dir(tmp_path: Path, qc=FP8_QC, tensors=None) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    cfg = {"model_type": "llama"}
    if qc is not None:
        cfg["quantization_config"] = qc
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if tensors is not None:
        _write_safetensors(d / "model.safetensors", tensors)
    return d


def _write_index(d: Path, weight_map: dict) -> None:
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")


def _torch(*, hip=None, cuda="12.4", caps=((9, 0),), current=0, xpu=None):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=SimpleNamespace(
            device_count=lambda: len(caps),
            current_device=lambda: current,
            get_device_capability=lambda i: caps[i],
            mem_get_info=lambda i: (4 * 10 ** 9, 8 * 10 ** 9)),
        xpu=xpu)


def _never_fetch():
    raise AssertionError("the Hub fetch policy was consulted")


# --------------------------------------------------------------------------
#  Native-FP8 capability
# --------------------------------------------------------------------------

def test_rocm_gfx1030_capability_10_3_is_not_fp8_capable():
    rocm = _torch(hip="7.13.0", cuda=None, caps=((10, 3),))
    assert _hf_fp8.native_blocker(rocm, "cuda", [0]) == "the GPU is an AMD ROCm device"


def test_rocm_is_not_fp8_capable_even_with_a_cuda_version_string():
    rocm = _torch(hip="7.13.0", cuda="12.4", caps=((10, 3),))
    assert _hf_fp8.native_blocker(rocm, "cuda", [0]) is not None


@pytest.mark.parametrize("cap,native", [((8, 6), False), ((8, 9), True),
                                        ((9, 0), True), ((7, 5), False)])
def test_nvidia_capability_threshold_is_8_9(cap, native):
    nv = _torch(caps=(cap,))
    assert (_hf_fp8.native_blocker(nv, "cuda", [0]) is None) is native


def test_every_device_in_the_map_must_be_fp8_capable():
    nv = _torch(caps=((9, 0), (8, 6)))
    assert _hf_fp8.native_blocker(nv, "cuda", [0, 1]) == \
        "GPU 1 has compute capability 8.6"
    assert _hf_fp8.native_blocker(nv, "cuda", [0]) is None


@pytest.mark.parametrize("device,where", [("cpu", "the CPU"), ("xpu", "an Intel XPU GPU"),
                                          ("mps", "the mps device")])
def test_non_cuda_devices_are_not_fp8_capable(device, where):
    assert _hf_fp8.native_blocker(_torch(), device, []) == f"the model runs on {where}"


def test_cuda_device_ids_follow_the_device_map():
    t = _torch(caps=((9, 0), (9, 0), (9, 0)))
    assert _hf_fp8.cuda_device_ids(t, {"device_map": {"": 2}}) == [2]
    assert _hf_fp8.cuda_device_ids(
        t, {"device_map": "auto", "max_memory": {1: 5, 0: 5, "cpu": 9}}) == [0, 1]
    assert _hf_fp8.cuda_device_ids(t, {"device_map": "auto"}) == [0, 1, 2]


def test_native_check_includes_the_current_cuda_device(tmp_path):
    d = _model_dir(tmp_path)
    nv = _torch(caps=((8, 6), (9, 0)), current=0)
    plan = _hf_fp8.plan_load(str(d), nv, "cuda",
                             {"device_map": "auto", "max_memory": {1: 5, "cpu": 9}},
                             hub_fetch_refusal=_never_fetch, triton_check=lambda: None)
    assert plan.native is False
    assert plan.reason == "GPU 0 has compute capability 8.6"


# --------------------------------------------------------------------------
#  Load plan
# --------------------------------------------------------------------------

def test_fp8_checkpoint_on_rocm_plans_bf16_expansion(tmp_path):
    d = _model_dir(tmp_path, tensors={"w": ("F8_E4M3", (256, 128)),
                                      "w_scale_inv": ("F32", (2, 1)),
                                      "n": ("BF16", (128,))})
    rocm = _torch(hip="7.13.0", cuda=None, caps=((10, 3),))
    plan = _hf_fp8.plan_load(str(d), rocm, "cuda", {"device_map": "auto"},
                             hub_fetch_refusal=_never_fetch,
                             triton_check=lambda: None)
    assert plan.native is False
    assert plan.reason == "the GPU is an AMD ROCm device"
    assert plan.expanded_bytes == (256 * 128 + 2 + 128) * 2


def test_fp8_checkpoint_on_cpu_plans_bf16_expansion(tmp_path):
    d = _model_dir(tmp_path)
    plan = _hf_fp8.plan_load(str(d), _torch(), "cpu", {"device_map": "cpu"},
                             hub_fetch_refusal=_never_fetch,
                             triton_check=lambda: None)
    assert plan.native is False
    assert plan.reason == "the model runs on the CPU"


def test_missing_triton_plans_bf16_expansion(tmp_path):
    d = _model_dir(tmp_path)
    plan = _hf_fp8.plan_load(str(d), _torch(), "cuda", {"device_map": "auto"},
                             hub_fetch_refusal=_never_fetch,
                             triton_check=lambda: "triton is not available")
    assert plan.native is False
    assert plan.reason == "triton is not available"


@pytest.mark.parametrize("refusal,offline", [(None, False), ("net_mode=off", True)])
def test_native_plan_is_offline_when_the_policy_refuses_the_hub(tmp_path, refusal, offline):
    d = _model_dir(tmp_path)
    plan = _hf_fp8.plan_load(str(d), _torch(), "cuda", {"device_map": "auto"},
                             hub_fetch_refusal=lambda: refusal,
                             triton_check=lambda: None)
    assert plan.native is True
    assert plan.offline is offline


@pytest.mark.parametrize("qc", [None, {"quant_method": "awq"},
                                {"quant_method": "fbgemm_fp8"}, "fp8"])
def test_other_checkpoints_get_no_fp8_plan(tmp_path, qc):
    d = _model_dir(tmp_path, qc=qc)
    assert _hf_fp8.plan_load(str(d), _torch(), "cpu", {"device_map": "cpu"}) is None


def test_quant_method_is_case_insensitive(tmp_path):
    d = _model_dir(tmp_path, qc={"quant_method": "FP8"})
    assert _hf_fp8.is_fp8_checkpoint(str(d))


# --------------------------------------------------------------------------
#  Expanded size
# --------------------------------------------------------------------------

def test_expanded_bytes_counts_integer_tensors_at_their_own_size(tmp_path):
    d = _model_dir(tmp_path, tensors={"ids": ("I64", (3,)), "m": ("BOOL", (5,)),
                                      "f": ("F32", (4,))})
    assert _hf_fp8.expanded_bf16_bytes(str(d)) == 3 * 8 + 5 + 4 * 2


def test_expanded_bytes_follows_the_shard_index_only(tmp_path):
    d = _model_dir(tmp_path)
    _write_safetensors(d / "model-00001-of-00002.safetensors", {"a": ("F8_E4M3", (10,))})
    _write_safetensors(d / "model-00002-of-00002.safetensors", {"b": ("BF16", (6,))})
    _write_safetensors(d / "consolidated.safetensors", {"a": ("F8_E4M3", (10,)),
                                                        "b": ("BF16", (6,))})
    _write_index(d, {"a": "model-00001-of-00002.safetensors",
                     "b": "model-00002-of-00002.safetensors"})
    assert _hf_fp8.expanded_bf16_bytes(str(d)) == (10 + 6) * 2


def test_single_file_ignores_other_safetensors_files(tmp_path):
    d = _model_dir(tmp_path, tensors={"a": ("F8_E4M3", (10,))})
    _write_safetensors(d / "consolidated.safetensors", {"a": ("F8_E4M3", (10,))})
    assert _hf_fp8.expanded_bf16_bytes(str(d)) == 10 * 2


def test_without_model_safetensors_or_index_there_is_no_size(tmp_path):
    d = _model_dir(tmp_path)
    _write_safetensors(d / "consolidated.safetensors", {"a": ("F8_E4M3", (10,))})
    assert _hf_fp8.expanded_bf16_bytes(str(d)) is None


@pytest.mark.parametrize("weight_map", [{"a": "../outside.safetensors"}, "not-a-map"])
def test_an_index_naming_a_file_outside_or_malformed_gives_no_size(tmp_path, weight_map):
    d = _model_dir(tmp_path, tensors={"a": ("F8_E4M3", (10,))})
    _write_safetensors(tmp_path / "outside.safetensors", {"a": ("F8_E4M3", (10,))})
    _write_index(d, weight_map)
    assert _hf_fp8.expanded_bf16_bytes(str(d)) is None


@pytest.mark.parametrize("payload", [
    b"\x01",
    struct.pack("<Q", 10 ** 12) + b"{}",
    struct.pack("<Q", 2) + b"[]",
    struct.pack("<Q", 5) + b"{bad}",
])
def test_unreadable_safetensors_header_gives_no_size(tmp_path, payload):
    d = _model_dir(tmp_path)
    (d / "model.safetensors").write_bytes(payload)
    assert _hf_fp8.expanded_bf16_bytes(str(d)) is None


def test_no_safetensors_gives_no_size(tmp_path):
    assert _hf_fp8.expanded_bf16_bytes(str(_model_dir(tmp_path))) is None


# --------------------------------------------------------------------------
#  Load-output wording
# --------------------------------------------------------------------------

def test_describe_states_the_expansion_and_the_reason():
    plan = _hf_fp8.Fp8Plan(native=False, reason="the model runs on the CPU",
                           expanded_bytes=1_503_318_528)
    assert _hf_fp8.describe(plan) == (
        "FP8 weights expanded to bf16 (about 2 bytes per parameter, about 1.5 GB); "
        "native FP8 is unavailable: the model runs on the CPU")


@pytest.mark.parametrize("offline,text", [
    (False, "FP8 weights run natively (finegrained-fp8 kernel)"),
    (True, "FP8 weights run natively (finegrained-fp8 kernel from the local cache)")])
def test_describe_a_native_load(offline, text):
    assert _hf_fp8.describe(_hf_fp8.Fp8Plan(native=True, offline=offline)) == text


def test_one_line_collapses_whitespace_and_cuts():
    assert _hf_fp8.one_line("a\n  b\tc") == "a b c"
    cut = _hf_fp8.one_line("x" * 500, limit=20)
    assert cut == "x" * 17 + "..." and len(cut) == 20


# --------------------------------------------------------------------------
#  Hub fetch policy
# --------------------------------------------------------------------------

def test_hub_fetch_is_refused_when_net_mode_is_off(monkeypatch):
    from localm.inference.backends import _hf_hub_gate
    monkeypatch.setenv("LOCALM_NET_MODE", "off")
    assert "net_mode=off" in _hf_hub_gate.hub_fetch_refusal()


def test_hub_fetch_checks_the_configured_endpoint(monkeypatch):
    from localm import netpolicy
    from localm.inference.backends import _hf_hub_gate
    seen = []
    monkeypatch.setattr(netpolicy, "check_url", seen.append)
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
    assert _hf_hub_gate.hub_fetch_refusal() is None
    monkeypatch.delenv("HF_ENDPOINT")
    assert _hf_hub_gate.hub_fetch_refusal() is None
    assert seen == ["https://hub.example.test", "https://huggingface.co"]


def test_hub_fetch_policy_error_counts_as_a_refusal(monkeypatch):
    from localm import netpolicy
    from localm.inference.backends import _hf_hub_gate

    def _boom(url):
        raise OSError("resolver exploded")

    monkeypatch.setattr(netpolicy, "check_url", _boom)
    assert _hf_hub_gate.hub_fetch_refusal() == (
        "the network policy could not be checked (OSError: resolver exploded)")


# --------------------------------------------------------------------------
#  Memory refusal
# --------------------------------------------------------------------------

def _ram(monkeypatch, n):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=n))


def test_memory_budget_cuda_and_cpu(monkeypatch):
    _ram(monkeypatch, 16 * 10 ** 9)
    t = _torch(caps=((9, 0), (9, 0)))
    assert _hf_fp8.memory_budget(t, "cuda", {"device_map": {"": 1}}) == 4 * 10 ** 9
    assert _hf_fp8.memory_budget(t, "cuda", {"device_map": "auto"}) == \
        16 * 10 ** 9 + 2 * 4 * 10 ** 9
    assert _hf_fp8.memory_budget(
        t, "cuda", {"device_map": "auto", "max_memory": {0: 3, "cpu": 7}}) == 10
    assert _hf_fp8.memory_budget(t, "cpu", {"device_map": "cpu"}) == 16 * 10 ** 9


def _xpu(free=None, total=None):
    def mem_get_info(i):
        if free is None:
            raise RuntimeError("not supported on this part")
        return free, total or free

    def get_device_properties(i):
        if total is None:
            raise RuntimeError("no properties")
        return SimpleNamespace(total_memory=total)

    return SimpleNamespace(mem_get_info=mem_get_info,
                           get_device_properties=get_device_properties)


@pytest.mark.parametrize("xpu,budget", [
    (_xpu(free=6 * 10 ** 9, total=8 * 10 ** 9), 6 * 10 ** 9),
    (_xpu(free=None, total=8 * 10 ** 9), 8 * 10 ** 9),
    (_xpu(free=None, total=None), 16 * 10 ** 9),
    (_xpu(free=40 * 10 ** 9, total=48 * 10 ** 9), 16 * 10 ** 9),
])
def test_memory_budget_xpu_is_the_smaller_of_ram_and_xpu_memory(monkeypatch, xpu, budget):
    _ram(monkeypatch, 16 * 10 ** 9)
    assert _hf_fp8.memory_budget(_torch(xpu=xpu), "xpu", {"device_map": "cpu"}) == budget


def test_too_big_expansion_is_refused_with_the_sizes(monkeypatch, tmp_path):
    _ram(monkeypatch, 150 * 10 ** 9)
    plan = _hf_fp8.Fp8Plan(native=False, reason="the model runs on the CPU",
                           expanded_bytes=320 * 10 ** 9)
    with pytest.raises(RuntimeError) as exc:
        _refuse_fp8_too_big(str(tmp_path / "Big-FP8"), plan, _torch(), "cpu",
                            {"device_map": "cpu"})
    msg = str(exc.value)
    assert "'Big-FP8' is an FP8 model" in msg
    assert "about 320.0 GB" in msg and "about 150.0 GB" in msg
    assert "the model runs on the CPU" in msg


def test_expansion_that_fits_is_not_refused(monkeypatch, tmp_path):
    _ram(monkeypatch, 150 * 10 ** 9)
    plan = _hf_fp8.Fp8Plan(native=False, reason="x", expanded_bytes=149 * 10 ** 9)
    _refuse_fp8_too_big(str(tmp_path), plan, _torch(), "cpu", {"device_map": "cpu"})


# --------------------------------------------------------------------------
#  Other Hub-kernel quantizations
# --------------------------------------------------------------------------

class _Dtype:
    def __init__(self, name, itemsize):
        self._name, self.itemsize = name, itemsize

    def __str__(self):
        return f"torch.{self._name}"


def _model(is_quantized, *dtypes):
    params = [SimpleNamespace(dtype=d, is_floating_point=lambda d=d: d is not None)
              for d in dtypes]
    return SimpleNamespace(is_quantized=is_quantized, parameters=lambda: iter(params))


@pytest.mark.parametrize("model,method,blocked,note", [
    (_model(False, None, _Dtype("bfloat16", 2)), "mxfp4", None,
     "MXFP4 weights expanded to bfloat16 (about 2 bytes per parameter)"),
    (_model(False, _Dtype("float32", 4)), "mxfp4",
     ("mxfp4", "Network access is disabled\n(net_mode=off)."),
     "MXFP4 weights expanded to float32 (about 4 bytes per parameter); "
     "the Hub kernel it needs is blocked: Network access is disabled (net_mode=off)."),
    (_model(False), "eetq", None, "EETQ weights expanded to full precision"),
    (_model(True, _Dtype("bfloat16", 2)), "mxfp4", None, None),
    (_model(False, _Dtype("bfloat16", 2)), "awq", None, None),
    (_model(False, _Dtype("bfloat16", 2)), None, None, None),
])
def test_dequantized_note(model, method, blocked, note):
    assert _dequantized_note(model, method, blocked) == note


def test_blocked_hub_kernel_message_names_the_method_and_the_policy(tmp_path):
    msg = _hub_kernel_blocked_message(str(tmp_path / "Tiny-EETQ"),
                                      ("eetq", "Network access is disabled."))
    assert msg == ("'Tiny-EETQ' uses eetq quantization, which needs a kernel downloaded "
                   "from the Hugging Face Hub, and the network policy does not allow "
                   "the download: Network access is disabled.")


# --------------------------------------------------------------------------
#  Load output
# --------------------------------------------------------------------------

def _load_output(monkeypatch, tmp_path, meta_extra) -> str:
    from rich.console import Console

    from localm.inference.backends import hf as hf_mod

    class _Runner:
        def spawn_and_load(self, params, timeout):
            return {"device": "cpu", "context_capacity": 64, **meta_extra}

        def is_alive(self):
            return True

    cap = Console(record=True, width=400, force_terminal=False, highlight=False)
    monkeypatch.setattr(hf_mod, "console", cap)
    monkeypatch.setattr(hf_mod, "HFRunner", _Runner)
    d = _model_dir(tmp_path)
    hf_mod.HFBackend(str(d)).load()
    return cap.export_text(styles=False)


def test_load_output_prints_each_worker_note(monkeypatch, tmp_path):
    note = ("FP8 weights expanded to bf16 (about 2 bytes per parameter, about "
            "1.5 GB); native FP8 is unavailable: the GPU is an AMD ROCm device")
    out = _load_output(monkeypatch, tmp_path, {"load_notes": [note, "second note"]})
    assert note in out
    assert "second note" in out
    assert out.index(note) < out.index("Model loaded (device: cpu)")


def test_load_output_notes_are_one_line_and_not_markup(monkeypatch, tmp_path):
    out = _load_output(monkeypatch, tmp_path,
                       {"load_notes": ["kernel failed: [bold]x[/bold]\n  trace line"]})
    assert "kernel failed: [bold]x[/bold] trace line" in out


@pytest.mark.parametrize("meta", [{}, {"load_notes": None}, {"load_notes": "text"}])
def test_load_output_without_notes(monkeypatch, tmp_path, meta):
    out = _load_output(monkeypatch, tmp_path, meta)
    assert out.strip().splitlines() == ["✓ Model loaded (device: cpu)"]


# --------------------------------------------------------------------------
#  Hub kernel gate (fresh interpreter each)
# --------------------------------------------------------------------------

def _run_child(tmp_path: Path, body: str, *, env_extra=None, fake_kernels=True) -> dict:
    """Run *body* in a fresh interpreter and return the JSON it prints last.
    With *fake_kernels* True, a ``kernels`` package that writes
    ``imported.txt`` when imported is first on ``sys.path``; a string is used
    as the rest of that package's ``__init__.py``."""
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    pkgs = tmp_path / "pkgs"
    marker = tmp_path / "imported.txt"
    if fake_kernels:
        (pkgs / "kernels").mkdir(parents=True, exist_ok=True)
        source = (fake_kernels if isinstance(fake_kernels, str) else
                  "def get_kernel(*a, **k):\n    return 'fake'\n")
        (pkgs / "kernels" / "__init__.py").write_text(
            f"open({str(marker)!r}, 'w').close()\n" + source, encoding="utf-8")
    script = scripts / "child.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_OFFLINE", "LOCALM_NET_MODE", "PYTHONPATH")}
    env.update({"LOCALM_HOME": str(home), "HF_HOME": str(tmp_path / "hf"),
                "PYTHONPATH": os.pathsep.join(
                    ([str(pkgs)] if fake_kernels else []) + [_REPO_ROOT])})
    env.update(env_extra or {})
    proc = subprocess.run([sys.executable, str(script)], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    result["marker"] = marker.exists()
    return result


_GATE_PROBE = """
    import importlib.util, json, os, sys
    from localm.inference.backends._hf_hub_gate import close_hub_gate
    before = importlib.util.find_spec("kernels") is not None
    close_hub_gate()
    after = importlib.util.find_spec("kernels") is not None
    try:
        from kernels import get_kernel
        imported = True
    except ImportError:
        imported = False
    print(json.dumps({"before": before, "after": after, "imported": imported,
                      "offline": os.environ.get("HF_HUB_OFFLINE")}))
"""


def test_closed_gate_hides_and_blocks_the_kernels_package(tmp_path):
    r = _run_child(tmp_path, _GATE_PROBE, env_extra={"LOCALM_NET_MODE": "ask"})
    assert r["before"] is True
    assert r["after"] is False
    assert r["imported"] is False
    assert r["marker"] is False


@pytest.mark.parametrize("mode,offline", [("off", "1"), ("ask", None), ("allow", None)])
def test_closed_gate_puts_the_hub_offline_only_when_net_mode_is_off(tmp_path, mode, offline):
    r = _run_child(tmp_path, _GATE_PROBE, env_extra={"LOCALM_NET_MODE": mode})
    assert r["offline"] == offline


def test_open_hub_kernels_restores_the_package(tmp_path):
    r = _run_child(tmp_path, """
        import json
        from localm.inference.backends import _hf_hub_gate as g
        g.close_hub_gate()
        blocked = g.hub_kernels_blocked()
        g.open_hub_kernels()
        from kernels import get_kernel
        print(json.dumps({"blocked": blocked, "after": g.hub_kernels_blocked(),
                          "value": get_kernel()}))
    """)
    assert r["blocked"] is True
    assert r["after"] is False
    assert r["value"] == "fake"
    assert r["marker"] is True


def test_gate_refuses_when_kernels_was_already_imported(tmp_path):
    r = _run_child(tmp_path, """
        import json, kernels
        from localm.inference.backends._hf_hub_gate import close_hub_gate
        try:
            close_hub_gate()
            err = None
        except RuntimeError as e:
            err = str(e)
        print(json.dumps({"err": err}))
    """)
    assert r["err"] and "kernels package was imported before" in r["err"]


def test_hf_worker_process_setup_closes_the_gate(tmp_path):
    r = _run_child(tmp_path, """
        import importlib.util, json
        from localm.inference.backends._hf_runner import prepare_worker_process
        prepare_worker_process()
        print(json.dumps({"found": importlib.util.find_spec("kernels") is not None}))
    """, env_extra={"LOCALM_NET_MODE": "ask"})
    assert r["found"] is False
    assert r["marker"] is False


@pytest.mark.parametrize("method,mode,opened", [
    ("mxfp4", "ask", True), ("mxfp4", "off", False), ("eetq", "allow", True),
    ("eetq", "off", False), ("awq", "allow", False), ("fp8", "allow", False)])
def test_hub_kernel_quantization_opens_the_gate_only_when_the_policy_allows(
        tmp_path, method, mode, opened):
    r = _run_child(tmp_path, f"""
        import importlib.util, json, os
        from localm.inference.backends._hf_hub_gate import close_hub_gate
        from localm.inference.backends._hf_worker import _open_gate_for_quant_method
        close_hub_gate()
        d = os.path.join(os.getcwd(), "m")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({{"quantization_config": {{"quant_method": {method!r}}}}}, f)
        blocked = _open_gate_for_quant_method(d)
        print(json.dumps({{"blocked": blocked,
                          "found": importlib.util.find_spec("kernels") is not None}}))
    """, env_extra={"LOCALM_NET_MODE": mode, "HF_ENDPOINT": "http://8.8.8.8"})
    assert r["found"] is opened
    if method in ("mxfp4", "eetq") and not opened:
        assert r["blocked"][0] == method
        assert "net_mode=off" in r["blocked"][1]
    else:
        assert r["blocked"] is None


def test_record_kernel_version_ref_makes_an_offline_version_lookup_work(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("kernels")
    hf = tmp_path / "hf"
    _fake_cached_kernel(hf)
    body = """
        import json, kernels
        from localm.inference.backends._hf_hub_gate import record_kernel_version_ref
        repo = "kernels-community/finegrained-fp8"
        try:
            kernels.get_kernel(repo, version=4)
            before = "loaded"
        except ValueError:
            before = "ValueError"
        kernels.get_kernel(repo, revision=%r)
        ref = record_kernel_version_ref(repo, 4)
        again = record_kernel_version_ref(repo, 4)
        print(json.dumps({"before": before, "ref": str(ref), "again": again}))
    """ % KERNEL_SHA
    env = {"HF_HUB_OFFLINE": "1"}
    r = _run_child(tmp_path, body, fake_kernels=False, env_extra=env)
    assert r["before"] == "ValueError"
    ref = Path(r["ref"])
    assert ref == hf / "hub" / "kernels--kernels-community--finegrained-fp8" / "refs" / "v4"
    assert ref.read_text(encoding="utf-8") == KERNEL_SHA
    assert r["again"] is None
    after = _run_child(tmp_path, """
        import json, kernels
        m = kernels.get_kernel("kernels-community/finegrained-fp8", version=4)
        print(json.dumps({"value": m.matmul_2d()}))
    """, fake_kernels=False, env_extra=env)
    assert after["value"] == "cached-kernel"


def test_record_kernel_version_ref_ignores_other_and_unknown_kernels(tmp_path):
    r = _run_child(tmp_path, """
        import json, types
        import kernels
        from localm.inference.backends._hf_hub_gate import record_kernel_version_ref
        mod = types.SimpleNamespace(__file__="/x/snapshots/abc/build/v/__init__.py")
        info = types.SimpleNamespace(repo_id="kernels-community/other", revision="abc")
        kernels.get_loaded_kernels = lambda: [
            types.SimpleNamespace(repo_info=info, module=mod),
            types.SimpleNamespace(repo_info=None, module=mod)]
        first = record_kernel_version_ref("kernels-community/finegrained-fp8", 4)
        no_version = record_kernel_version_ref("kernels-community/other", None)
        stale = types.SimpleNamespace(repo_id="kernels-community/other", revision="zzz")
        kernels.get_loaded_kernels = lambda: [
            types.SimpleNamespace(repo_info=stale, module=mod)]
        mismatch = record_kernel_version_ref("kernels-community/other", 4)
        print(json.dumps({"other": first, "no_version": no_version,
                          "mismatch": mismatch}))
    """)
    assert r["other"] is None
    assert r["no_version"] is None
    assert r["mismatch"] is None


def _fake_cached_kernel(hf: Path) -> Path:
    """A finegrained-fp8 kernel snapshot in the Hub cache under *hf*, as a
    by-commit fetch leaves it (no refs)."""
    variant = (hf / "hub" / "kernels--kernels-community--finegrained-fp8" / "snapshots"
               / KERNEL_SHA / "build" / "torch-universal")
    variant.mkdir(parents=True)
    (variant / "metadata.json").write_text(json.dumps({
        "name": "finegrained-fp8", "id": "_finegrained_fp8_localm_test", "version": 4,
        "license": "apache-2.0", "python-depends": [], "backend": {"type": "cpu"}}),
        encoding="utf-8")
    (variant / "__init__.py").write_text(
        "def matmul_2d(*a, **k):\n    return 'cached-kernel'\n"
        "matmul_batched = matmul_grouped = matmul_2d\n", encoding="utf-8")
    return variant


# --------------------------------------------------------------------------
#  Real transformers: tiny checkpoint, CPU
# --------------------------------------------------------------------------

_AUDIT = """
    import sys
    events = []
    def _audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            events.append(event)
        elif event == "import" and args[0] == "kernels":
            events.append("import kernels")
    sys.addaudithook(_audit)

    import json, os
    from localm.inference.backends._hf_runner import prepare_worker_process
    prepare_worker_process()
"""

_BUILD_TINY = """
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    model_dir = os.path.join(os.getcwd(), "tiny-model")
    os.makedirs(model_dir)
    H, I, V = 128, 256, 16
    torch.manual_seed(0)
    tensors = {
        "model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(V, H, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
    }
    reference = {}
    shapes = {"self_attn.q_proj": (H, H), "self_attn.k_proj": (H, H),
              "self_attn.v_proj": (H, H), "self_attn.o_proj": (H, H),
              "mlp.gate_proj": (I, H), "mlp.up_proj": (I, H), "mlp.down_proj": (H, I)}
    for name, (o, i) in shapes.items():
        w = torch.randn(o, i) * 0.05
        blocks = w.reshape(o // 128, 128, i // 128, 128)
        scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0
        q = (blocks / scale[:, None, :, None]).reshape(o, i).to(torch.float8_e4m3fn)
        key = f"model.layers.0.{name}.weight"
        tensors[key] = q
        tensors[f"model.layers.0.{name}.weight_scale_inv"] = scale.float()
        reference[key] = (q.float().reshape(o // 128, 128, i // 128, 128)
                          * scale[:, None, :, None]).reshape(o, i)
    save_file(tensors, os.path.join(model_dir, "model.safetensors"),
              metadata={"format": "pt"})
    qc = json.loads(os.environ.get("TINY_QC") or json.dumps(
        {"quant_method": "fp8", "activation_scheme": "dynamic",
         "weight_block_size": [128, 128]}))
    config = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": V, "hidden_size": H, "intermediate_size": I,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
        "max_position_embeddings": 64, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "bos_token_id": 1, "eos_token_id": 2, "quantization_config": qc,
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"w{n}": n for n in range(3, V)})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(os.path.join(model_dir, "tokenizer.json"))
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",
                   "bos_token": "<s>", "eos_token": "</s>"}, f)

    import dataclasses
    load_calls = []
    if os.environ.get("SPY_LOAD") == "1":
        import transformers
        for _name in ("AutoModelForImageTextToText", "AutoModelForSeq2SeqLM",
                      "AutoModelForCausalLM", "AutoModel"):
            _cls = getattr(transformers, _name, None)
            if _cls is None:
                continue
            def _spy(cls, *a, _orig=_cls.from_pretrained, **k):
                qc_arg = k.get("quantization_config")
                load_calls.append({"type": type(qc_arg).__name__,
                                   "dequantize": getattr(qc_arg, "dequantize", None)})
                return _orig(*a, **k)
            _cls.from_pretrained = classmethod(_spy)
"""

_REAL_LOAD = _AUDIT + _BUILD_TINY + """
    from localm.inference.backends._hf_worker import HFWorker
    worker = HFWorker(model_dir, device="cpu")
    worker.load()
    model = worker._model
    params = dict(model.named_parameters())
    fp8_params = [n for n, p in params.items() if p.dtype == torch.float8_e4m3fn]
    max_err = 0.0
    for key, ref in reference.items():
        got = params[key].detach().float()
        max_err = max(max_err, float(((got - ref).abs() / (ref.abs() + 1e-3)).max()))
    with torch.no_grad():
        logits = model(torch.tensor([[3, 4, 5]])).logits
    finite = bool(torch.isfinite(logits).all())

    import huggingface_hub.constants as hub_constants
    from transformers.integrations import hub_kernels
    lazy = hub_kernels.lazy_load_kernel("finegrained-fp8", mapping={})
    try:
        hub_kernels.get_kernel("kernels-community/finegrained-fp8", version=4)
        get_kernel_error = None
    except ImportError as e:
        get_kernel_error = type(e).__name__
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel
    try:
        load_finegrained_fp8_kernel()
        fp8_kernel_error = None
    except ImportError as e:
        fp8_kernel_error = type(e).__name__

    print(json.dumps({
        "plan": dataclasses.asdict(worker.fp8_plan), "notes": worker.load_notes,
        "load_calls": load_calls, "fp8_params": fp8_params,
        "max_rel_err": max_err, "finite": finite,
        "linear_dtype": str(params["model.layers.0.mlp.down_proj.weight"].dtype),
        "lazy_load_kernel": repr(lazy), "get_kernel_error": get_kernel_error,
        "fp8_kernel_error": fp8_kernel_error,
        "kernels_module": repr(sys.modules.get("kernels", "absent")),
        "hub_offline": bool(hub_constants.HF_HUB_OFFLINE),
        "events": events,
        "kernel_cache": [p for p in os.listdir(os.environ["HF_HOME"])
                         if "kernels" in p] if os.path.isdir(os.environ["HF_HOME"]) else [],
    }))
"""


def _needs_real_stack():
    for mod in ("torch", "transformers", "safetensors", "tokenizers"):
        pytest.importorskip(mod)


@pytest.fixture(scope="module")
def real_fp8_load(tmp_path_factory):
    _needs_real_stack()
    tmp = tmp_path_factory.mktemp("real_fp8")
    return _run_child(tmp, _REAL_LOAD, fake_kernels=False, env_extra={
        "LOCALM_NET_MODE": "off", "HF_ENDPOINT": "http://127.0.0.1:9", "SPY_LOAD": "1"})


def test_real_fp8_checkpoint_loads_dequantized_to_bf16_on_cpu(real_fp8_load):
    r = real_fp8_load
    assert r["plan"]["native"] is False
    assert r["plan"]["reason"] == "the model runs on the CPU"
    assert r["fp8_params"] == []
    assert r["linear_dtype"] == "torch.bfloat16"
    assert r["max_rel_err"] < 0.02
    assert r["finite"] is True
    assert r["notes"] == [_hf_fp8.describe(_hf_fp8.Fp8Plan(**r["plan"]))]


def test_real_fp8_load_passes_an_explicit_dequantize_config(real_fp8_load):
    calls = real_fp8_load["load_calls"]
    assert calls
    assert all(c == {"type": "FineGrainedFP8Config", "dequantize": True} for c in calls)


def test_real_net_off_load_never_reaches_a_hub_kernel(real_fp8_load):
    r = real_fp8_load
    assert r["events"] == []
    assert r["kernels_module"] == "None"
    assert r["hub_offline"] is True
    assert r["lazy_load_kernel"] == "None"
    assert r["get_kernel_error"] == "ImportError"
    assert r["fp8_kernel_error"] == "ImportError"
    assert r["kernel_cache"] == []


_LOAD_EXPECTING_ERROR = _AUDIT + _BUILD_TINY + """
    from localm.inference.backends._hf_worker import HFWorker
    try:
        HFWorker(model_dir, device="cpu").load()
        err = None
    except RuntimeError as e:
        err = str(e)
    print(json.dumps({"err": err, "events": events}))
"""


def test_real_eetq_load_with_the_hub_refused_names_the_policy(tmp_path):
    _needs_real_stack()
    r = _run_child(tmp_path, _LOAD_EXPECTING_ERROR, fake_kernels=False, env_extra={
        "LOCALM_NET_MODE": "off", "HF_ENDPOINT": "http://127.0.0.1:9",
        "TINY_QC": json.dumps({"quant_method": "eetq", "weights": "int8"})})
    assert r["err"] is not None
    assert r["err"].startswith(
        "'tiny-model' uses eetq quantization, which needs a kernel downloaded from the "
        "Hugging Face Hub, and the network policy does not allow the download: ")
    assert "net_mode=off" in r["err"]
    assert r["events"] == []


# --------------------------------------------------------------------------
#  Real transformers: native plan through the reopened gate
# --------------------------------------------------------------------------

_FAKE_KERNELS = '''
import contextlib, os, types

class _Dummy:
    def __init__(self, *a, **k):
        pass

class Mode:
    INFERENCE = 1
    TRAINING = 2
    TORCH_COMPILE = 4

def use_kernel_forward_from_hub(*a, **k):
    return lambda cls: cls

def use_kernelized_func(*a, **k):
    return lambda cls: cls

@contextlib.contextmanager
def use_kernel_mapping(*a, **k):
    yield

def get_loaded_kernels():
    return []

def get_kernel(repo_id, **kwargs):
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    with open(os.environ["FAKE_KERNEL_CALLS"], "a") as f:
        f.write(repo_id + ("@offline" if offline else "") + " ")
    mode = os.environ.get("FAKE_KERNEL_MODE", "")
    if mode == "fail" or (mode == "cache-only" and not offline):
        raise ModuleNotFoundError("No module named 'triton'")
    m = types.ModuleType("fake_finegrained_fp8")
    m.matmul_2d = m.matmul_batched = m.matmul_grouped = lambda *a, **k: None
    return m

def __getattr__(name):
    return _Dummy
'''

_PATCH_NATIVE = """
    from localm.inference.backends import _hf_fp8, _hf_hub_gate
    _hf_fp8.native_blocker = lambda *a, **k: None
    _hf_fp8.triton_blocker = lambda: None
"""

_NATIVE_LOAD = _AUDIT + _PATCH_NATIVE + """
    if os.environ.get("FAKE_POLICY") == "allow":
        _hf_hub_gate.hub_fetch_refusal = lambda: None
""" + _BUILD_TINY + """
    from localm.inference.backends._hf_worker import HFWorker
    worker = HFWorker(model_dir, device="cpu")
    worker.load()
    params = dict(worker._model.named_parameters())
    print(json.dumps({
        "plan": dataclasses.asdict(worker.fp8_plan), "notes": worker.load_notes,
        "fp8_params": sum(p.dtype == torch.float8_e4m3fn for p in params.values()),
        "accelerator": bool(torch.cuda.is_available()),
        "kernels_module": type(sys.modules.get("kernels")).__name__,
        "calls": open(os.environ["FAKE_KERNEL_CALLS"]).read().split(),
    }))
"""

_REPO = "kernels-community/finegrained-fp8"


@pytest.mark.parametrize("mode,calls,native,offline", [
    ("", [_REPO], True, False),
    ("cache-only", [_REPO, _REPO + "@offline"], True, True),
    ("fail", [_REPO, _REPO + "@offline"], False, False),
])
def test_native_fp8_plan_loads_the_kernel_through_the_reopened_gate(
        tmp_path, mode, calls, native, offline):
    _needs_real_stack()
    calls_file = tmp_path / "calls.txt"
    calls_file.write_text("", encoding="utf-8")
    env = {"LOCALM_NET_MODE": "ask", "FAKE_POLICY": "allow",
           "HF_ENDPOINT": "http://127.0.0.1:9", "FAKE_KERNEL_CALLS": str(calls_file),
           "FAKE_KERNEL_MODE": mode}
    r = _run_child(tmp_path, _NATIVE_LOAD, fake_kernels=_FAKE_KERNELS, env_extra=env)
    assert r["calls"] == calls
    assert r["kernels_module"] == "module"
    assert r["plan"]["native"] is native
    assert r["plan"]["offline"] is offline
    assert r["notes"] == [_hf_fp8.describe(_hf_fp8.Fp8Plan(**r["plan"]))]
    if native:
        if r["accelerator"]:
            assert r["fp8_params"] > 0
    else:
        assert r["plan"]["reason"] == (
            "the finegrained-fp8 kernel could not be loaded "
            "(ModuleNotFoundError: No module named 'triton')")
        assert r["fp8_params"] == 0


def test_real_kernels_offline_native_load_uses_the_recorded_cache(tmp_path):
    _needs_real_stack()
    pytest.importorskip("kernels")
    hf = tmp_path / "hf"
    _fake_cached_kernel(hf)
    refs = hf / "hub" / "kernels--kernels-community--finegrained-fp8" / "refs"
    refs.mkdir()
    (refs / "v4").write_text(KERNEL_SHA, encoding="utf-8")
    r = _run_child(tmp_path, _NATIVE_LOAD.replace(
        '"calls": open(os.environ["FAKE_KERNEL_CALLS"]).read().split(),',
        '"calls": [], "events": events,'), fake_kernels=False, env_extra={
        "LOCALM_NET_MODE": "off", "HF_ENDPOINT": "http://127.0.0.1:9"})
    assert r["plan"]["native"] is True
    assert r["plan"]["offline"] is True
    assert r["notes"] == [
        "FP8 weights run natively (finegrained-fp8 kernel from the local cache)"]
    assert r["kernels_module"] == "module"
    assert [e for e in r["events"] if e.startswith("socket")] == []
