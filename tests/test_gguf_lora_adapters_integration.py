# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end test of GGUF LoRA adapters: a tiny base model and a published
LoRA adapter for it, driven through the native llama runtime.

No mocks. Downloads SmolLM2-135M-Instruct (about 100 MB at Q4_K_M) and a 5 MB
adapter made for it, and checks that

* the adapter changes what the model generates (the same prompt without it
  generates something else);
* localm's first 20 greedy tokens with the adapter equal what upstream's own
  completion program generates with ``--lora`` (when the runtime ships it);
* an adapter that matches the base's architecture but not its tensor shapes is
  refused by the native loader and the reason reaches the caller;
* an adapter attached in the registry is applied by a real engine load, through
  the isolated worker process, and reported as applied.

@integration so the default ``pytest -m "not integration"`` skips it: it needs the
native runtime provisioned and the files fetched on first run. CPU only.
"""

from __future__ import annotations

import pytest

from tests._lora_oracle import upstream_completion_dll, upstream_completion_text
from tests._real_gguf import fetch_gguf, require_native_runtime

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_BASE_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
_BASE_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
_ADAPTER_REPO = "unileon-robotics/SmolLM2-135M-Instruct-BehaviorTree-LoRA-GGUF"
_ADAPTER_FILE = "f16.gguf"
_FOREIGN_REPO = "garamu/TinyLlama-1.1B-Chat-v1.0-lora-F16-GGUF"
_FOREIGN_FILE = "TinyLlama-1.1B-Chat-v1.0-lora-f16.gguf"

_PROMPT = "Write a behavior tree for a robot that picks up a cup:\n"
_N = 20


@pytest.fixture(scope="module")
def base_gguf():
    require_native_runtime()
    return fetch_gguf(_BASE_REPO, _BASE_FILE)


@pytest.fixture(scope="module")
def adapter_gguf():
    require_native_runtime()
    return fetch_gguf(_ADAPTER_REPO, _ADAPTER_FILE)


@pytest.fixture(scope="module")
def foreign_adapter_gguf():
    require_native_runtime()
    return fetch_gguf(_FOREIGN_REPO, _FOREIGN_FILE)


def _same_generation(upstream: str, local: str) -> bool:
    """True when *upstream* is exactly *local* followed by nothing but the
    whitespace upstream's program prints after its last generated token."""
    return upstream.startswith(local) and not upstream[len(local):].strip()


def _localm_greedy(base, adapters=None):
    """localm's first _N greedy tokens after _PROMPT, and their text, from an
    in-process LlamaCpp loaded with *adapters*."""
    from localm.inference.backends.llamacpp.llama import LlamaCpp
    llm = LlamaCpp(base, n_ctx=512, n_gpu_layers=0, verbose=False, n_threads=2,
                   adapters=adapters)
    try:
        prompt = llm.tokenize(_PROMPT, add_bos=True)
        tokens = list(llm._generate(
            prompt, max_new_tokens=_N, temperature=0.0, top_k=1, top_p=1.0,
            repeat_penalty=1.0, seed=1))
        return tokens, llm.detokenize(tokens), list(llm.applied_adapters)
    finally:
        llm.close()


def test_the_adapter_changes_the_output(base_gguf, adapter_gguf):
    plain, plain_text, plain_applied = _localm_greedy(base_gguf)
    tuned, tuned_text, applied = _localm_greedy(base_gguf, [(adapter_gguf, 1.0)])
    assert len(plain) == len(tuned) == _N
    assert plain_applied == []
    assert applied == [{"path": adapter_gguf, "scale": 1.0}]
    assert tuned != plain, (plain_text, tuned_text)


def test_the_scale_changes_the_output(base_gguf, adapter_gguf):
    full, _, _ = _localm_greedy(base_gguf, [(adapter_gguf, 1.0)])
    faint, _, _ = _localm_greedy(base_gguf, [(adapter_gguf, 0.05)])
    assert full != faint


@pytest.mark.skipif(upstream_completion_dll() is None,
                    reason="the provisioned runtime has no upstream completion library")
def test_the_first_20_greedy_tokens_equal_upstreams(base_gguf, adapter_gguf):
    plain, plain_text, _ = _localm_greedy(base_gguf)
    tuned, tuned_text, _ = _localm_greedy(base_gguf, [(adapter_gguf, 1.0)])

    upstream_plain = upstream_completion_text(base_gguf, _PROMPT, _N)
    upstream_tuned = upstream_completion_text(
        base_gguf, _PROMPT, _N, adapter=adapter_gguf)

    # The control: without an adapter the two programs agree, so a difference
    # with one is the adapter's doing and not a harness difference.
    assert _same_generation(upstream_plain, plain_text), (upstream_plain, plain_text)
    assert _same_generation(upstream_tuned, tuned_text), (upstream_tuned, tuned_text)
    assert upstream_tuned != upstream_plain


def test_an_adapter_made_for_other_tensors_is_refused_with_the_native_reason(
        base_gguf, foreign_adapter_gguf):
    """TinyLlama and SmolLM2 are both the llama architecture, so the header check
    passes and the native loader is the one that refuses the tensor shapes."""
    from localm.inference.backends.base import AdapterLoadError
    from localm.inference.backends.gguf import GgufBackend
    backend = GgufBackend(base_gguf, n_ctx=512, n_gpu_layers=0,
                          adapters=[(foreign_adapter_gguf, 1.0)])
    try:
        with pytest.raises(AdapterLoadError) as ei:
            backend.load()
    finally:
        backend.unload()
    message = str(ei.value)
    assert _FOREIGN_FILE in message
    assert "shape" in message or "base model" in message or "does not exist" in message, message
    assert "localm setup-llama" not in message


def test_a_registry_attachment_is_applied_by_an_engine_load_in_the_worker(
        base_gguf, adapter_gguf, tmp_path, monkeypatch):
    import shutil

    import localm.config as cfg
    import localm.model_manager as mm
    from localm.inference.engine import Engine

    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")

    base_copy = tmp_path / "files" / "base.gguf"
    adapter_copy = tmp_path / "files" / "tuned.gguf"
    base_copy.parent.mkdir()
    shutil.copy(base_gguf, base_copy)
    shutil.copy(adapter_gguf, adapter_copy)
    assert mm.add_local(str(base_copy), name="base") is True
    assert mm.add_local(str(adapter_copy), name="tuned") is True
    assert mm.load_registry()["tuned"]["model_type"] == "lora"
    mm.attach_adapter("tuned", "base", 1.0)

    messages = [{"role": "user", "content": "Write a behavior tree for a robot."}]

    def _reply(engine):
        return "".join(engine.chat_stream(
            messages, max_tokens=24, temperature=0.0, top_k=1, top_p=1.0,
            repeat_penalty=1.0, seed=1))

    engine = Engine(str(base_copy), display_name="base", n_ctx=512, n_gpu_layers=0)
    try:
        engine.load()
        assert engine.applied_adapters == [{"name": "tuned.gguf", "scale": 1.0}]
        with_adapter = _reply(engine)
        engine.unload()

        assert mm.detach_adapter("tuned") is True
        engine.load()
        assert engine.applied_adapters == []
        without_adapter = _reply(engine)
    finally:
        engine.unload()
    assert with_adapter and without_adapter
    assert with_adapter != without_adapter
