# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/model_census.py reads each model's own metadata, derives the role it
implies, and fails every model localm types differently or as unknown."""
import importlib.util
import json
import pickle
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "model_census.py"
_spec = importlib.util.spec_from_file_location("model_census", _SCRIPT)
mc = importlib.util.module_from_spec(_spec)
sys.modules["model_census"] = mc
_spec.loader.exec_module(mc)


def _gstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _gval(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", 7) + struct.pack("<?", v)
    if isinstance(v, int):
        return struct.pack("<I", 4) + struct.pack("<I", v)
    if isinstance(v, float):
        return struct.pack("<I", 6) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + _gstr(v)
    if isinstance(v, list):
        return struct.pack("<I", 9) + struct.pack("<IQ", 8, len(v)) + b"".join(_gstr(x) for x in v)
    raise TypeError(v)


def write_gguf(path: Path, kv: dict, tensors=("token_embd.weight", "output.weight"),
               truncate: int = 0) -> Path:
    """A GGUF v3 file with *kv* and one 8-element f32 tensor per name, its data
    aligned to 32 bytes; *truncate* cuts that many bytes off the end."""
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", len(tensors), len(kv))]
    for k, v in kv.items():
        out.append(_gstr(k) + _gval(v))
    for i, name in enumerate(tensors):
        out.append(_gstr(name) + struct.pack("<I", 1) + struct.pack("<Q", 8)
                   + struct.pack("<I", 0) + struct.pack("<Q", i * 32))
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    data = head + b"\0" * (32 * len(tensors))
    path.write_bytes(data[:len(data) - truncate] if truncate else data)
    return path


def write_safetensors(path: Path, names, metadata=None) -> Path:
    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for n in names}
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")
    return path


def write_folder(path: Path, config: dict, extra=()) -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in extra:
        (path / name).write_text("{}", encoding="utf-8")
    return path


def llm_kv(arch="qwen3"):
    return {"general.architecture": arch, "general.type": "model", f"{arch}.block_count": 2,
            "tokenizer.ggml.model": "gpt2"}


@pytest.mark.parametrize("kv,tensors,kind", [
    (llm_kv(), None, "llm"),
    ({**llm_kv(), "qwen3.pooling_type": 4}, None, "reranker"),
    ({**llm_kv("bert"), "bert.pooling_type": 2}, None, "embedding"),
    ({"general.architecture": "gemma-embedding", "gemma-embedding.block_count": 2}, None, "embedding"),
    ({"general.architecture": "t5encoder", "t5encoder.block_count": 24,
      "tokenizer.ggml.model": "t5"}, ["enc.blk.0.attn_k.weight"], "text-encoder"),
    ({"general.architecture": "clip", "general.type": "mmproj", "clip.has_vision_encoder": True},
     ["v.blk.0.attn_out.weight"], "projector"),
    ({"general.architecture": "llama", "general.type": "adapter", "adapter.type": "lora"},
     ["blk.0.attn_q.weight.lora_a"], "lora"),
    (llm_kv("qwen3tts"), None, "tts"),
    ({"general.architecture": "qwen_image21"}, ["img_in.weight", "transformer_blocks.0.attn.to_q.weight"],
     "diffusion"),
    ({}, ["cond_stage_model.transformer.text_model.final_layer_norm.bias",
          "model.diffusion_model.input_blocks.0.0.weight", "first_stage_model.decoder.conv_in.weight"],
     "diffusion"),
    ({"general.architecture": "mystery"}, ["x.weight"], "undetermined"),
])
def test_gguf_implied_kind(tmp_path, kv, tensors, kind):
    path = write_gguf(tmp_path / "m.gguf", kv, tensors or ("token_embd.weight", "output.weight"))
    implied = mc.implied_kind(path)
    assert implied.kind == kind, implied.evidence
    assert implied.architecture == kv.get("general.architecture")


@pytest.mark.parametrize("names,meta,kind", [
    (["double_blocks.0.processor.proj_lora1.down.weight"], {}, "lora"),
    (["lora_unet_down_blocks_0.lora_down.weight"], {}, "lora"),
    (["model.diffusion_model.blocks.0.cross_attn.k.weight"], {}, "diffusion"),
    (["blocks.0.self_attn.q.weight", "patch_embedding.weight"], {}, "diffusion"),
    (["conv1.weight", "decoder.conv1.weight", "encoder.conv1.weight"], {}, "vae"),
    (["decoder.conv1.weight"], {"modelspec.architecture": "qwen_image_2.1_vae"}, "vae"),
    (["encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight"], {}, "text-encoder"),
    (["text_model.encoder.layers.0.mlp.fc1.weight"], {}, "text-encoder"),
    (["control_model.input_blocks.0.0.weight"], {}, "controlnet"),
    (["conv_first.weight", "body.0.rdb1.conv1.weight"], {}, "upscaler"),
    (["emb_params"], {}, "textual-inversion"),
    (["something.weight"], {}, "undetermined"),
])
def test_safetensors_implied_kind(tmp_path, names, meta, kind):
    path = write_safetensors(tmp_path / "m.safetensors", names, meta)
    assert mc.implied_kind(path).kind == kind


def test_pth_upscaler_read_without_unpickling(tmp_path):
    path = tmp_path / "RealESRGAN_x4plus.pth"
    payload = pickle.dumps({"params_ema": {"conv_first.weight": 0, "body.0.rdb1.conv1.weight": 0}})
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("RealESRGAN_x4plus/data.pkl", payload)
    assert mc.implied_kind(path).kind == "upscaler"


def test_pth_pickle_never_executes(tmp_path):
    marker = tmp_path / "ran"

    class Boom:
        def __reduce__(self):
            return (Path.touch, (marker,))

    path = tmp_path / "evil.pth"
    path.write_bytes(pickle.dumps({"conv_first.weight": Boom()}))
    mc.implied_kind(path)
    assert not marker.exists()


@pytest.mark.parametrize("config,extra,kind", [
    ({"_class_name": "ACEStepTransformer2DModel"}, (), "diffusion"),
    ({"_class_name": "AutoencoderDC"}, (), "vae"),
    ({"_class_name": "ADaMoSHiFiGANV1"}, (), "vocoder"),
    ({"architectures": ["UMT5EncoderModel"], "model_type": "umt5"}, (), "text-encoder"),
    ({"architectures": ["QWenLMHeadModel"]}, (), "llm"),
    ({"architectures": ["BertModel"], "model_type": "bert"}, ("modules.json",), "embedding"),
    ({"architectures": ["XLMRobertaForSequenceClassification"], "id2label": {"0": "LABEL_0"}},
     (), "reranker"),
    ({"architectures": ["LlamaForCausalLM"]}, ("adapter_config.json",), "lora"),
    ({"architectures": ["Mystery"]}, (), "undetermined"),
])
def test_folder_implied_kind(tmp_path, config, extra, kind):
    assert mc.implied_kind(write_folder(tmp_path / "m", config, extra)).kind == kind


@pytest.mark.parametrize("data", [b"", b"GGUF", b"NOPE" + b"\0" * 40,
                                  b"GGUF" + struct.pack("<IQQ", 3, 1, 10 ** 9)])
def test_corrupt_gguf_is_undetermined(tmp_path, data):
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)
    implied = mc.implied_kind(path)
    assert implied.kind == "undetermined" and "unreadable" in implied.evidence


def test_enumerate_skips_hidden_and_partial_and_groups_splits(tmp_path):
    (tmp_path / ".cache").mkdir()
    write_gguf(tmp_path / "a.gguf", llm_kv())
    (tmp_path / "b.gguf.part").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    write_gguf(tmp_path / "big-00002-of-00002.gguf", {"split.no": 1})
    write_gguf(tmp_path / "big-00001-of-00002.gguf", llm_kv())
    write_folder(tmp_path / "hf", {"architectures": ["QWenLMHeadModel"]})
    nested = tmp_path / "loras"
    nested.mkdir()
    write_safetensors(nested / "x.safetensors", ["a.lora_down.weight"])
    entries, skipped = mc.enumerate_models(tmp_path)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in (e.path for e in entries)) == [
        "a.gguf", "big-00001-of-00002.gguf", "hf", "loras/x.safetensors"]
    big = next(e for e in entries if e.path.name.startswith("big"))
    assert len(big.parts) == 2
    assert sorted((p.name, why) for p, why in skipped) == [
        (".cache", "hidden"), ("b.gguf.part", "partial download"),
        ("notes.txt", "not a model file suffix (.txt)")]


def _imp(kind, arch=None):
    return mc.Implied(kind, "test", arch)


@pytest.mark.parametrize("implied,entry,api,verdict", [
    (_imp("llm", "qwen3"), {"model_type": "llm", "architecture": "qwen3"}, {"model_type": "llm"}, "PASS"),
    (_imp("text-encoder", "t5encoder"), {"model_type": "embedding", "architecture": "t5encoder"},
     {"model_type": "embedding"}, "MISMATCH"),
    (_imp("vocoder"), {"model_type": "unknown"}, {"model_type": "unknown"}, "UNKNOWN"),
    (_imp("reranker", "qwen3"), {"model_type": "embedding", "architecture": "qwen3", "reranker": True},
     {"model_type": "embedding"}, "PASS"),
    (_imp("reranker", "qwen3"), {"model_type": "embedding", "architecture": "qwen3"},
     {"model_type": "embedding"}, "MISMATCH"),
    (_imp("embedding", "bert"), {"model_type": "embedding", "architecture": "bert", "reranker": True},
     {"model_type": "embedding"}, "MISMATCH"),
    (_imp("llm", "qwen3"), {"model_type": "llm", "architecture": "llama"}, {"model_type": "llm"}, "MISMATCH"),
    (_imp("llm"), {"model_type": "llm"}, None, "NOT-LISTED"),
    (_imp("llm"), {"model_type": "llm"}, {"model_type": "embedding"}, "MISMATCH"),
    (_imp("upscaler"), None, None, "NOT-REGISTERED"),
    (_imp("undetermined"), {"model_type": "llm"}, {"model_type": "llm"}, "UNDETERMINED"),
])
def test_judge(implied, entry, api, verdict):
    row = mc.judge("m", implied, "m" if entry else None, entry, api, "exit 0: Not a model")
    assert row.verdict == verdict, row.problems
    assert (verdict == "PASS") == (row.problems == [])


def test_metadata_only_exit_codes(tmp_path, capsys):
    write_gguf(tmp_path / "a.gguf", llm_kv())
    assert mc.main(["--models-dir", str(tmp_path), "--metadata-only"]) == 0
    write_safetensors(tmp_path / "odd.safetensors", ["something.weight"])
    assert mc.main(["--models-dir", str(tmp_path), "--metadata-only"]) == 1
    assert "1 undetermined" in capsys.readouterr().out


def test_cannot_run_is_exit_2(tmp_path, capsys):
    assert mc.main(["--models-dir", str(tmp_path / "absent"), "--metadata-only"]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert mc.main(["--models-dir", str(empty), "--metadata-only"]) == 2
    write_gguf(empty / "a.gguf", llm_kv())
    assert mc.main(["--models-dir", str(empty)]) == 2
    assert "--work-dir is required" in capsys.readouterr().out


def test_census_rows_from_a_registry(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "chat.gguf", llm_kv())
    write_gguf(models / "enc.gguf", {"general.architecture": "t5encoder", "t5encoder.block_count": 1,
                                     "tokenizer.ggml.model": "t5"}, ["enc.blk.0.attn_k.weight"])
    write_safetensors(models / "up.safetensors", ["conv_first.weight", "body.0.rdb1.conv1.weight"])
    calls = []

    def fake_register(python, root, entries, home, timeout):
        calls.append(json.loads((home / "config.json").read_text(encoding="utf-8")))
        (home / "registry.json").write_text(json.dumps({
            "chat": {"path": str(models / "chat.gguf"), "model_type": "llm", "architecture": "qwen3"},
            "enc": {"path": str(models / "enc.gguf"), "model_type": "embedding", "architecture": "t5encoder"},
        }), encoding="utf-8")
        return {mc._norm(models / "up.safetensors"): "exit 0: Not a model"}

    monkeypatch.setattr(mc, "register_all", fake_register)
    monkeypatch.setattr(mc, "fetch_api_models", lambda python, home, timeout, info: [
        {"name": "chat", "model_type": "llm"}, {"name": "enc", "model_type": "embedding"}])
    monkeypatch.setattr(mc, "localm_identity", lambda python, home: "test")
    rows, info = mc.census(models, sys.executable, tmp_path / "work", 60, False, False)
    assert {r.entry: r.verdict for r in rows} == {
        "chat.gguf": "PASS", "enc.gguf": "MISMATCH", "up.safetensors": "NOT-REGISTERED"}
    assert calls == [{"update_url": "", "bugreport_upload_url": ""}]
    assert info["home_deleted"] is True and not Path(info["home"]).exists()


def test_truncated_gguf_is_undetermined(tmp_path):
    whole = write_gguf(tmp_path / "whole.gguf", llm_kv())
    assert mc.implied_kind(whole).kind == "llm"
    cut = write_gguf(tmp_path / "cut.gguf", llm_kv(), truncate=1)
    implied = mc.implied_kind(cut)
    assert implied.kind == "undetermined" and "truncated" in implied.evidence


def test_other_files_are_listed_as_skipped(tmp_path):
    write_gguf(tmp_path / "ok.gguf", llm_kv())
    (tmp_path / "tiny.llamafile").write_bytes(b"MZ")
    (tmp_path / "carrier.exe").write_bytes(b"MZ")
    entries, skipped = mc.enumerate_models(tmp_path)
    assert sorted(e.path.name for e in entries) == ["ok.gguf", "tiny.llamafile"]
    assert [(p.name, why) for p, why in skipped] == [("carrier.exe", "not a model file suffix (.exe)")]
    assert mc.implied_kind(tmp_path / "tiny.llamafile").kind == "undetermined"


def _stub_census(monkeypatch, models, leftover=None, fail=False):
    def fake_register(python, root, entries, home, timeout):
        (home / "registry.json").write_text(json.dumps({
            "chat": {"path": str(models / "chat.gguf"), "model_type": "llm", "architecture": "qwen3"}}))
        return {}

    def fake_fetch(python, home, timeout, info):
        if leftover:
            info["leftover_pids"] = leftover
        if fail:
            raise RuntimeError("server did not answer")
        return [{"name": "chat", "model_type": "llm"}]

    monkeypatch.setattr(mc, "register_all", fake_register)
    monkeypatch.setattr(mc, "fetch_api_models", fake_fetch)
    monkeypatch.setattr(mc, "localm_identity", lambda python, home: "test")


def test_leftovers_fail_the_run(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "chat.gguf", llm_kv())
    work = tmp_path / "work"
    _stub_census(monkeypatch, models)
    assert mc.main(["--models-dir", str(models), "--work-dir", str(work)]) == 0
    monkeypatch.setattr(mc.shutil, "rmtree", lambda path, ignore_errors=False: None)
    assert mc.main(["--models-dir", str(models), "--work-dir", str(work)]) == 2
    assert "left its throwaway home behind" in capsys.readouterr().out
    _stub_census(monkeypatch, models, leftover=[4242])
    assert mc.main(["--models-dir", str(models), "--work-dir", str(work), "--keep"]) == 2
    assert "could not stop server process(es): [4242]" in capsys.readouterr().out


def test_home_deletion_is_reported_after_a_failed_run(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "chat.gguf", llm_kv())
    _stub_census(monkeypatch, models, fail=True)
    monkeypatch.setattr(mc.shutil, "rmtree", lambda path, ignore_errors=False: None)
    assert mc.main(["--models-dir", str(models), "--work-dir", str(tmp_path / "work")]) == 2
    out = capsys.readouterr().out
    assert "could not run: server did not answer" in out
    assert "left its throwaway home behind" in out


def test_kill_tree_stops_grandchildren(tmp_path):
    psutil = pytest.importorskip("psutil")
    child = ("import subprocess, sys, time\n"
             "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],"
             " creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n"
             "print(p.pid, flush=True)\n"
             "time.sleep(120)\n")
    import subprocess as sp
    proc = sp.Popen([sys.executable, "-c", child], stdout=sp.PIPE, text=True,
                    creationflags=getattr(sp, "CREATE_NO_WINDOW", 0) | getattr(sp, "CREATE_NEW_PROCESS_GROUP", 0),
                    start_new_session=sys.platform != "win32")
    grandchild = int(proc.stdout.readline())
    seen = mc._descendants(proc.pid)
    assert grandchild in seen
    assert mc._kill_tree(proc, seen) == []
    assert proc.poll() is not None
    assert not psutil.pid_exists(grandchild) or \
        psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE


@pytest.mark.integration
def test_real_localm_add_and_api_models(tmp_path):
    """The census against the real ``localm add`` and a real GUI server: a chat,
    an embedding and a reranker GGUF all pass, two of them being the same size,
    and a loose file localm refuses is reported as not registered."""
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "tiny-chat.gguf", llm_kv("llama"),
               ("token_embd.weight", "output.weight", "blk.0.attn_q.weight"))
    write_gguf(models / "tiny-embed.gguf", {**llm_kv("bert"), "bert.pooling_type": 2})
    write_gguf(models / "tiny-reranker.gguf", {**llm_kv(), "qwen3.pooling_type": 4})
    assert (models / "tiny-embed.gguf").stat().st_size == (models / "tiny-reranker.gguf").stat().st_size
    (models / "refused.pth").write_bytes(pickle.dumps({"text_model.encoder.layers.0.mlp.fc1.weight": 0}))
    rows, info = mc.census(models, sys.executable, tmp_path / "work", 300, False, False)
    verdicts = {r.entry: (r.verdict, r.registry_type, r.api_type) for r in rows}
    assert verdicts["tiny-chat.gguf"] == ("PASS", "llm", "llm"), rows
    assert verdicts["tiny-embed.gguf"] == ("PASS", "embedding", "embedding"), rows
    assert verdicts["tiny-reranker.gguf"][0] == "PASS", rows
    assert verdicts["refused.pth"][0] == "NOT-REGISTERED", rows
    assert info["home_deleted"] is True
    shutil.rmtree(tmp_path / "work", ignore_errors=True)
