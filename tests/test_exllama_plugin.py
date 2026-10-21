# SPDX-License-Identifier: AGPL-3.0-or-later
"""The exllama plugin (EXL2 / EXL3 through ExLlama), tested without CUDA.

The ExLlama runtimes, torch and transformers are replaced by small fake packages
written into tmp_path; they run in this process for the worker's own logic and
in a real spawned worker for the load / stream / cancel / crash / unload path.
"""

from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

from localm.inference.backends.base import (
    ChatTemplateMissingError,
    ContextCapacityExceededError,
    GrammarUnsupportedError,
    ModelRefusedError,
    UnsupportedInputError,
)
from localm.model_manager.unsupported import hf_folder_refusal
from localm.plugins.builtin.exllama import backend as exl_backend
from localm.plugins.builtin.exllama import plug as exl_plug
from localm.plugins.builtin.exllama import worker as exl_worker

# Verbatim quantization_config blocks from the public repositories named.
EXL2_QUANT = json.loads(
    '{"quant_method":"exl2","version":"0.1.7","bits":4.0,"head_bits":6,'
    '"calibration":{"rows":115,"length":2048,"dataset":"(default)"}}')  # turboderp/...-exl2 @4.0bpw
EXL3_QUANT = json.loads(
    '{"quant_method":"exl3","version":"0.0.0","bits":4.0,'
    '"calibration":{"rows":100,"cols":2048}}')  # turboderp/...-exl3 @4.0bpw
AWQ_QUANT = json.loads(
    '{"bits":4,"group_size":128,"modules_to_not_convert":null,"quant_method":"awq",'
    '"version":"gemm","zero_point":true}')  # Qwen/Qwen2.5-0.5B-Instruct-AWQ
LLAMA_SHAPE = {"architectures": ["LlamaForCausalLM"], "hidden_size": 2048,
               "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 16,
               "max_position_embeddings": 131072, "eos_token_id": [128001, 128009]}


def _model_dir(d: Path, quant: dict | None, extra: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    config = dict(LLAMA_SHAPE, **(extra or {}))
    if quant is not None:
        config["quantization_config"] = quant
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * 1000)
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "generation_config.json").write_text('{"eos_token_id": 128008}', encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
#  claims
# --------------------------------------------------------------------------- #

class TestClaims:
    @pytest.mark.parametrize("quant,fmt", [
        (EXL2_QUANT, "exl2"), (EXL3_QUANT, "exl3"),
        (dict(EXL2_QUANT, quant_method="EXL2"), "exl2"),
    ])
    def test_exllama_folders_are_claimed(self, tmp_path, quant, fmt):
        d = _model_dir(tmp_path / "m", quant)
        assert exl_plug.quant_format(str(d)) == fmt
        assert exl_plug.claims(str(d)) is True

    @pytest.mark.parametrize("quant", [
        None, AWQ_QUANT, {"quant_method": "gptq", "bits": 4},
        {"quant_method": 2}, {"quant_method": "exl2", "load_in_4bit": True},
        {"quant_method": " exl2"}, "exl2",
    ])
    def test_other_folders_are_not(self, tmp_path, quant):
        d = _model_dir(tmp_path / "m", quant)
        assert exl_plug.claims(str(d)) is False

    def test_unreadable_configs_and_files_are_not_claimed(self, tmp_path):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "config.json").write_text("{not json", encoding="utf-8")
        listed = tmp_path / "listed"
        listed.mkdir()
        (listed / "config.json").write_text("[1]", encoding="utf-8")
        f = tmp_path / "m.gguf"
        f.write_bytes(b"GGUF")
        for p in (bad, listed, f, tmp_path / "missing"):
            assert exl_plug.claims(str(p)) is False

    @pytest.mark.parametrize("quant", [
        EXL2_QUANT, EXL3_QUANT, dict(EXL2_QUANT, quant_method="EXL2"), AWQ_QUANT, None,
        {"quant_method": "exl2", "load_in_8bit": True}, {"quant_method": " exl2"},
    ])
    def test_claimed_exactly_when_core_names_exllama(self, tmp_path, quant):
        d = _model_dir(tmp_path / "m", quant)
        refusal = hf_folder_refusal(d) or ""
        assert exl_plug.claims(str(d)) == ("only ExLlama" in refusal)


# --------------------------------------------------------------------------- #
#  refusals in the parent, before any worker exists
# --------------------------------------------------------------------------- #

class _Detection:
    def __init__(self, vendors, state):
        self.vendors, self.gpu_state = vendors, state


@pytest.fixture
def no_spawn(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a worker was started")
    monkeypatch.setattr(exl_backend, "HFRunner", refuse)


class TestParentRefusals:
    @pytest.mark.parametrize("vendors,state,have", [
        (["amd"], "found", "an AMD GPU"), (["intel"], "found", "an Intel GPU"),
        (["amd", "intel"], "found", "AMD and Intel GPUs"), ([], "none", "no GPU"),
    ])
    def test_no_nvidia_gpu_is_refused(self, tmp_path, monkeypatch, no_spawn,
                                      vendors, state, have):
        monkeypatch.setattr("localm.hwdetect.detect", lambda: _Detection(vendors, state))
        b = exl_backend.ExllamaBackend(str(_model_dir(tmp_path / "m", EXL2_QUANT)), fmt="exl2")
        with pytest.raises(ModelRefusedError) as exc:
            b.load()
        assert f"this machine has {have}." in str(exc.value)
        assert "EXL2" in str(exc.value) and "NVIDIA" in str(exc.value)
        assert not b.loaded

    def test_non_linux_is_refused(self, tmp_path, monkeypatch, no_spawn):
        monkeypatch.setattr("localm.hwdetect.detect", lambda: _Detection(["nvidia"], "found"))
        monkeypatch.setattr(exl_backend.sys, "platform", "win32")
        b = exl_backend.ExllamaBackend(str(_model_dir(tmp_path / "m", EXL3_QUANT)), fmt="exl3")
        with pytest.raises(ModelRefusedError, match="on Linux with an NVIDIA GPU only"):
            b.load()

    def test_missing_runtime_is_refused_with_the_install_command(
            self, tmp_path, monkeypatch, no_spawn):
        monkeypatch.setattr("localm.hwdetect.detect", lambda: _Detection(["nvidia"], "found"))
        monkeypatch.setattr(exl_backend.sys, "platform", "linux")
        monkeypatch.setattr(exl_backend.importlib.util, "find_spec",
                            lambda name: None if name == "exllamav3" else object())
        b = exl_backend.ExllamaBackend(str(_model_dir(tmp_path / "m", EXL3_QUANT)), fmt="exl3")
        with pytest.raises(ModelRefusedError) as exc:
            b.load()
        assert "exllamav3 is not installed" in str(exc.value)
        assert "localm plugin install-deps exllama" in str(exc.value)

    def test_an_unanswered_gpu_probe_is_left_to_the_worker(self, monkeypatch):
        monkeypatch.setattr("localm.hwdetect.detect", lambda: _Detection([], "unknown"))
        assert exl_backend._gpu_refusal("exl2") is None

    def test_this_machine_refuses_unless_it_has_an_nvidia_gpu(self, tmp_path, no_spawn):
        from localm.hwdetect import detect
        if "nvidia" in detect().vendors:
            pytest.skip("this machine has an NVIDIA GPU")
        b = exl_backend.ExllamaBackend(str(_model_dir(tmp_path / "m", EXL2_QUANT)), fmt="exl2")
        with pytest.raises(ModelRefusedError):
            b.load()

    def test_images_and_grammars_are_refused_before_the_worker(self, tmp_path):
        b = exl_backend.ExllamaBackend(str(_model_dir(tmp_path / "m", EXL2_QUANT)), fmt="exl2")
        image = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]
        with pytest.raises(UnsupportedInputError):
            list(b.chat_stream(image))
        with pytest.raises(GrammarUnsupportedError):
            list(b.chat_stream([{"role": "user", "content": "hi"}], grammar="root ::= \"a\""))
        with pytest.raises(GrammarUnsupportedError):
            b.validate_grammar("root ::= \"a\"")


# --------------------------------------------------------------------------- #
#  VRAM sizing
# --------------------------------------------------------------------------- #

class TestVramSizing:
    def test_cache_tokens_round_up_to_a_page_and_cap_at_the_model(self, tmp_path):
        d = _model_dir(tmp_path / "m", EXL2_QUANT, {"max_position_embeddings": 3000})
        assert exl_backend.cache_tokens_for(str(d), 1000) == 1024
        assert exl_backend.cache_tokens_for(str(d), 1024) == 1024
        assert exl_backend.cache_tokens_for(str(d), 100000) == 3072

    def test_cache_tokens_default_to_the_n_ctx_setting(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.config.load_config", lambda: {"n_ctx": 5000})
        d = _model_dir(tmp_path / "m", EXL2_QUANT)
        assert exl_backend.cache_tokens_for(str(d), None) == 5120

    def test_full_offload_need_is_weights_plus_fp16_cache(self, tmp_path):
        from localm.inference.residency import model_footprint_bytes
        d = _model_dir(tmp_path / "m", EXL2_QUANT)
        b = exl_backend.ExllamaBackend(str(d), n_ctx=4096, fmt="exl2")
        # 4096 tokens x 16 layers x 8 kv heads x 64 head dim x (K, V) x 2 bytes
        assert b.full_offload_vram_bytes() == (
            model_footprint_bytes(d) + 4096 * 16 * 8 * 64 * 2 * 2)

    def test_need_is_unknown_without_the_attention_shape(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"quantization_config": EXL2_QUANT}))
        assert exl_backend.ExllamaBackend(str(d), n_ctx=4096).full_offload_vram_bytes() is None

    def test_a_nested_text_config_is_read(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "quantization_config": EXL3_QUANT,
            "text_config": {"num_hidden_layers": 2, "num_key_value_heads": 1,
                            "head_dim": 128, "num_attention_heads": 4}}))
        assert exl_backend.kv_cache_bytes(str(d), 256) == 256 * 2 * 1 * 128 * 2 * 2


# --------------------------------------------------------------------------- #
#  fake runtimes
# --------------------------------------------------------------------------- #

FAKES = {
    "torch/__init__.py": '''
        import os

        class _Version:
            hip = os.environ.get("FAKE_TORCH_HIP") or None
            cuda = "12.8"

        version = _Version()

        class _Cuda:
            @staticmethod
            def is_available():
                return os.environ.get("FAKE_TORCH_CUDA", "1") == "1"

            @staticmethod
            def device_count():
                return 2

            @staticmethod
            def get_device_name(i):
                return f"Fake NVIDIA {i}"

            @staticmethod
            def empty_cache():
                pass

        cuda = _Cuda()
        long = "long"

        class Tensor:
            def __init__(self, data):
                self.data = data
                self.shape = (len(data), len(data[0]) if data else 0)

            def to(self, device):
                return self

        def tensor(data, dtype=None):
            return Tensor(data)

        def ones_like(t):
            return Tensor([[1] * t.shape[-1]])
    ''',
    "transformers/__init__.py": '''
        class _Encoded(dict):
            def to(self, device):
                return self

        class _Tokenizer:
            eos_token_id = 2

            def __init__(self, template):
                self.chat_template = template

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True, **kwargs):
                out = []
                for m in messages:
                    c = m["content"]
                    if not isinstance(c, str):
                        c = "".join(p["text"] for p in c)
                    out.append(f"<{m['role']}>{c}")
                return "".join(out) + ("<assistant>" if add_generation_prompt else "")

            def encode(self, text, add_special_tokens=False):
                return [ord(ch) for ch in text]

            def __call__(self, text, return_tensors=None, add_special_tokens=False,
                         split_special_tokens=False):
                ids = [ord(ch) for ch in text]
                if return_tensors == "pt":
                    import torch
                    return _Encoded(input_ids=torch.tensor([ids]))
                return {"input_ids": ids}

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, trust_remote_code=None):
                import os
                if trust_remote_code is not False:
                    raise AssertionError("trust_remote_code must be False")
                return _Tokenizer(os.environ.get("FAKE_CHAT_TEMPLATE", "{{ messages }}"))
    ''',
    "_fake_exl.py": '''
        import json
        import os
        from pathlib import Path

        REPLY = ["Hello", " there", "!"]

        def log(event, **kw):
            path = os.environ.get("FAKE_EXL_LOG")
            if path:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": event, **kw}) + "\\n")

        def mode():
            path = os.environ.get("FAKE_EXL_MODE_FILE")
            if path and Path(path).is_file():
                return Path(path).read_text().strip()
            return "reply"

        class Generator:
            def __init__(self, **kw):
                self.jobs = []

            def enqueue(self, job):
                job.step = 0
                self.jobs.append(job)

            def num_remaining_jobs(self):
                return len(self.jobs)

            def cancel(self, job):
                log("cancel")
                if job in self.jobs:
                    self.jobs.remove(job)

            def iterate(self):
                job = self.jobs[0]
                m = mode()
                job.step += 1
                if job.step == 1:
                    return [{"job": job, "stage": "prefill", "eos": False}]
                n = job.step - 1
                if m == "crash" and n == 2:
                    os._exit(3)
                if m == "endless":
                    import time
                    time.sleep(0.02)
                    return [{"job": job, "stage": "streaming", "eos": False, "text": "x"}]
                last_word = n >= len(REPLY)
                last_budget = n >= job.max_new_tokens
                r = {"job": job, "stage": "streaming", "eos": last_word or last_budget,
                     "text": REPLY[n - 1]}
                if r["eos"]:
                    r["eos_reason"] = "max_new_tokens" if last_budget and not last_word else "stop_token"
                    self.jobs.remove(job)
                return [r]

        class Job:
            def __init__(self, input_ids, max_new_tokens, seed=None, stop_conditions=None,
                         sampler=None, gen_settings=None):
                self.max_new_tokens = max_new_tokens
                sampling = sampler.kw if sampler is not None else dict(vars(gen_settings))
                log("job", n_prompt=input_ids.shape[-1], max_new_tokens=max_new_tokens,
                    seed=seed, stop=list(stop_conditions or []), sampling=sampling)
    ''',
    "exllamav3/__init__.py": '''
        from _fake_exl import Generator as _Gen, Job, log

        class Config:
            @staticmethod
            def from_directory(d):
                log("v3_config", path=d)
                return Config()

        class Model:
            @staticmethod
            def from_config(c):
                return Model()

            def load(self, device=None, **kw):
                log("v3_load", device=device)

            def unload(self):
                log("v3_unload")

        class Cache:
            def __init__(self, model, max_num_tokens):
                log("v3_cache", tokens=max_num_tokens)

        class Tokenizer:
            @staticmethod
            def from_config(c):
                return Tokenizer()

        class Generator(_Gen):
            pass
    ''',
    "exllamav3/generator/__init__.py": "",
    "exllamav3/generator/sampler.py": '''
        class ComboSampler:
            def __init__(self, **kw):
                self.kw = kw
    ''',
    "exllamav2/__init__.py": '''
        from _fake_exl import log

        class ExLlamaV2Config:
            def __init__(self, path):
                log("v2_config", path=path)
                self.max_seq_len = 999999

            def arch_compat_overrides(self):
                log("v2_compat")

        class ExLlamaV2:
            def __init__(self, config):
                self.config = config

            def load_autosplit(self, cache):
                log("v2_load", max_seq_len=self.config.max_seq_len)

            def unload(self):
                log("v2_unload")

        class ExLlamaV2Cache:
            def __init__(self, model, max_seq_len, lazy):
                log("v2_cache", tokens=max_seq_len, lazy=lazy)

        class ExLlamaV2Tokenizer:
            def __init__(self, config):
                pass
    ''',
    "exllamav2/generator.py": '''
        from _fake_exl import Generator as ExLlamaV2DynamicGenerator
        from _fake_exl import Job as ExLlamaV2DynamicJob

        class ExLlamaV2Sampler:
            class Settings:
                pass
    ''',
}

_FAKE_NAMES = ("torch", "transformers", "_fake_exl", "exllamav3", "exllamav3.generator",
               "exllamav3.generator.sampler", "exllamav2", "exllamav2.generator")


def _write_fakes(root: Path) -> Path:
    for rel, body in FAKES.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(textwrap.dedent(body), encoding="utf-8")
    return root


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """The fake packages on sys.path (for a spawned worker) and, for this
    process, under their real names in sys.modules until the test ends."""
    root = _write_fakes(tmp_path / "fakes")
    log = tmp_path / "exl.log"
    mode = tmp_path / "mode.txt"
    monkeypatch.setenv("FAKE_EXL_LOG", str(log))
    monkeypatch.setenv("FAKE_EXL_MODE_FILE", str(mode))
    saved = {n: sys.modules.get(n) for n in _FAKE_NAMES}
    for n in _FAKE_NAMES:
        sys.modules.pop(n, None)
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()

    class Env:
        def events(self):
            if not log.is_file():
                return []
            return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

        def set_mode(self, value):
            mode.write_text(value, encoding="utf-8")

    yield Env()
    for n in _FAKE_NAMES:
        if saved[n] is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = saved[n]


def _worker(d: Path, fmt: str, cache_tokens: int = 1024, device=None):
    w = exl_worker.ExllamaWorker(model_path=str(d), fmt=fmt, cache_tokens=cache_tokens,
                                 device=device)
    w.load()
    return w


HI = [{"role": "user", "content": "hi"}]


class TestWorkerInProcess:
    def test_exl3_loads_streams_and_unloads(self, tmp_path, fakes):
        d = _model_dir(tmp_path / "m", EXL3_QUANT)
        w = _worker(d, "exl3", cache_tokens=2048, device="cuda:1")
        assert w.context_capacity == 2048 and w.resolved_device == "cuda:1"
        statuses = []
        out = "".join(w.chat_stream(HI, max_tokens=64, temperature=0.5, top_p=0.9, top_k=20,
                                    repeat_penalty=1.05, min_p=0.1, presence_penalty=0.2,
                                    frequency_penalty=0.3, seed=7, on_status=statuses.append))
        assert out == "Hello there!"
        assert w.last_finish_reason == "stop"
        assert statuses == ["Processing prompt..."]
        w.unload()
        events = fakes.events()
        assert {"event": "v3_cache", "tokens": 2048} in events
        assert {"event": "v3_load", "device": "cuda:1"} in events
        job = next(e for e in events if e["event"] == "job")
        assert job["seed"] == 7 and job["n_prompt"] == len("<user>hi<assistant>")
        assert job["stop"] == [2, 128001, 128008, 128009]
        assert job["sampling"] == {"rep_p": 1.05, "pres_p": 0.2, "freq_p": 0.3,
                                   "temperature": 0.5, "min_p": 0.1, "top_k": 20, "top_p": 0.9}
        assert events[-1] == {"event": "v3_unload"}
        assert not w.loaded

    def test_exl2_uses_exllamav2_across_every_gpu(self, tmp_path, fakes):
        d = _model_dir(tmp_path / "m", EXL2_QUANT)
        w = _worker(d, "exl2", cache_tokens=4096, device="cuda:0")
        assert w.resolved_device == "cuda:0"
        assert any("ExLlamaV2 splits EXL2" in n for n in w.load_notes)
        assert "".join(w.chat_stream(HI, max_tokens=64)) == "Hello there!"
        events = fakes.events()
        assert {"event": "v2_cache", "tokens": 4096, "lazy": True} in events
        assert {"event": "v2_load", "max_seq_len": 4096} in events
        job = next(e for e in events if e["event"] == "job")
        assert job["sampling"]["token_repetition_penalty"] == 1.1
        assert job["sampling"]["token_presence_penalty"] == 0.0

    def test_the_token_budget_ends_with_length(self, tmp_path, fakes):
        w = _worker(_model_dir(tmp_path / "m", EXL3_QUANT), "exl3")
        assert "".join(w.chat_stream(HI, max_tokens=2)) == "Hello there"
        assert w.last_finish_reason == "length"

    def test_unlimited_tokens_resolve_to_the_room_left(self, tmp_path, fakes):
        w = _worker(_model_dir(tmp_path / "m", EXL3_QUANT), "exl3", cache_tokens=256)
        list(w.chat_stream(HI, max_tokens=0))
        job = next(e for e in fakes.events() if e["event"] == "job")
        assert job["max_new_tokens"] == 256 - len("<user>hi<assistant>")

    def test_a_cancel_stops_the_job(self, tmp_path, fakes):
        import threading
        fakes.set_mode("endless")
        w = _worker(_model_dir(tmp_path / "m", EXL3_QUANT), "exl3")
        cancel = threading.Event()
        got = []
        for piece in w.chat_stream(HI, max_tokens=500, cancel_event=cancel):
            got.append(piece)
            if len(got) == 3:
                cancel.set()
        assert got == ["x", "x", "x"]
        assert {"event": "cancel"} in fakes.events()
        assert w.last_finish_reason == "stop"

    def test_a_prompt_that_fills_the_cache_is_refused(self, tmp_path, fakes):
        w = _worker(_model_dir(tmp_path / "m", EXL3_QUANT), "exl3", cache_tokens=10)
        with pytest.raises(ContextCapacityExceededError):
            list(w.chat_stream(HI))
        assert not any(e["event"] == "job" for e in fakes.events())

    def test_media_parts_and_grammars_are_refused(self, tmp_path, fakes):
        w = _worker(_model_dir(tmp_path / "m", EXL3_QUANT), "exl3")
        audio = [{"role": "user", "content": [{"type": "text", "text": "a"},
                                              {"type": "input_audio", "input_audio": {}}]}]
        with pytest.raises(UnsupportedInputError):
            list(w.chat_stream(audio))
        with pytest.raises(GrammarUnsupportedError):
            list(w.chat_stream(HI, grammar="root ::= \"a\""))

    def test_text_parts_render_through_the_template(self, tmp_path, fakes):
        w = _worker(_model_dir(tmp_path / "m", EXL3_QUANT), "exl3")
        parts = [{"role": "user", "content": [{"type": "text", "text": "a"},
                                              {"type": "text", "text": "b"}]}]
        assert w.count_messages_tokens(parts) == len("<user>ab<assistant>")
        assert w.count_tokens("abc") == 3

    def test_a_missing_chat_template_is_refused(self, tmp_path, fakes, monkeypatch):
        monkeypatch.setenv("FAKE_CHAT_TEMPLATE", "")
        w = _worker(_model_dir(tmp_path / "m", EXL3_QUANT), "exl3")
        with pytest.raises(ChatTemplateMissingError):
            list(w.chat_stream(HI))

    @pytest.mark.parametrize("env,why", [
        ({"FAKE_TORCH_HIP": "7.1"}, "is a ROCm build"),
        ({"FAKE_TORCH_CUDA": "0"}, "sees no CUDA device"),
    ])
    def test_no_cuda_device_is_a_refusal(self, tmp_path, monkeypatch, env, why):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        root = _write_fakes(tmp_path / "fakes")
        saved = {n: sys.modules.get(n) for n in _FAKE_NAMES}
        try:
            for n in _FAKE_NAMES:
                sys.modules.pop(n, None)
            monkeypatch.syspath_prepend(str(root))
            importlib.invalidate_caches()
            w = exl_worker.ExllamaWorker(model_path=str(_model_dir(tmp_path / "m", EXL3_QUANT)),
                                         fmt="exl3", cache_tokens=256)
            with pytest.raises(ModelRefusedError, match=why):
                w.load()
        finally:
            for n in _FAKE_NAMES:
                if saved[n] is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = saved[n]

    def test_an_unknown_format_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            exl_worker.ExllamaWorker(model_path=str(tmp_path), fmt="gptq", cache_tokens=256)


# --------------------------------------------------------------------------- #
#  a real spawned worker
# --------------------------------------------------------------------------- #

@pytest.fixture
def backend_env(fakes, monkeypatch):
    for name in ("_gpu_refusal", "_platform_refusal", "_runtime_refusal"):
        monkeypatch.setattr(exl_backend, name, lambda fmt: None)
    return fakes


class TestRealWorkerProcess:
    def test_load_stream_cancel_and_unload(self, tmp_path, backend_env):
        d = _model_dir(tmp_path / "m", EXL3_QUANT)
        b = exl_backend.ExllamaBackend(str(d), n_ctx=1000, fmt="exl3")
        try:
            b.load()
            assert b.loaded and b.effective_ctx_max == 1024 and b.n_ctx_max == 1024
            assert "".join(b.chat_stream(HI, max_tokens=64)) == "Hello there!"
            assert b.last_finish_reason == "stop"
            assert b.count_tokens("abcd") == 4

            backend_env.set_mode("endless")
            stream = b.chat_stream(HI, max_tokens=500)
            assert [next(stream) for _ in range(3)] == ["x", "x", "x"]
            stream.close()
            assert b.loaded, "a cancelled stream must keep the worker"
            backend_env.set_mode("reply")
            assert "".join(b.chat_stream(HI, max_tokens=1)) == "Hello"
            assert b.last_finish_reason == "length"
        finally:
            b.unload()
        assert not b.loaded
        events = [e["event"] for e in backend_env.events()]
        assert "cancel" in events
        assert events[-1] == "v3_unload"

    def test_a_worker_crash_mid_stream_is_reported_and_unloads(self, tmp_path, backend_env):
        d = _model_dir(tmp_path / "m", EXL2_QUANT)
        b = exl_backend.ExllamaBackend(str(d), n_ctx=512, fmt="exl2")
        try:
            b.load()
            backend_env.set_mode("crash")
            with pytest.raises(RuntimeError):
                "".join(b.chat_stream(HI, max_tokens=64))
            assert not b.loaded
            with pytest.raises(RuntimeError, match="not loaded"):
                list(b.chat_stream(HI))
        finally:
            b.unload()

    def test_the_plugin_creates_this_backend_for_an_exllama_folder(self, tmp_path):
        d = _model_dir(tmp_path / "m", EXL2_QUANT)
        made = exl_plug.create(str(d), device=None, n_ctx=2048)
        assert isinstance(made, exl_backend.ExllamaBackend)
        assert made._fmt == "exl2" and made._n_ctx == 2048
