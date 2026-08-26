# SPDX-License-Identifier: AGPL-3.0-or-later
"""The launcher's model selector must list ONLY LLM (chat) models.

A non-LLM registry entry (an embedding / text-encoder / LoRA / VAE / diffusion
component, or an unclassified 'unknown') is not something you launch as the chat
model, so it must never appear in the launcher dropdown.
"""

import importlib.machinery
import importlib.util
from pathlib import Path

_LAUNCHER = Path(__file__).resolve().parents[1] / "launcher.pyw"


def _load_launcher():
    # .pyw is a recognized Python source suffix only on Windows; pass an explicit
    # SourceFileLoader so launcher.pyw loads as source on every platform.
    loader = importlib.machinery.SourceFileLoader("localm_launcher_mod", str(_LAUNCHER))
    spec = importlib.util.spec_from_file_location("localm_launcher_mod", _LAUNCHER, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MIXED_REGISTRY = {
    "chat-llm":       {"model_type": "llm", "path": "/x/chat.gguf"},
    "legacy-no-type": {"path": "/x/legacy.gguf"},                 # legacy -> treated as llm
    "an-embedding":   {"model_type": "text-encoder", "path": "/x/bge"},
    "a-lora":         {"model_type": "lora", "path": "/x/lora"},
    "a-vae":          {"model_type": "vae", "path": "/x/vae"},
    "a-diffusion":    {"model_type": "diffusion-unet", "path": "/x/unet"},
    "a-mmproj":       {"model_type": "mmproj", "path": "/x/mmproj"},
    "unclassified":   {"model_type": "unknown", "path": "/x/mystery"},
}


def test_launcher_lists_only_llm_models(monkeypatch):
    import localm.config as cfg
    import localm.model_manager as mm
    monkeypatch.setattr(cfg, "load_registry", lambda: dict(_MIXED_REGISTRY))
    monkeypatch.setattr(mm, "sync_models_dir", lambda *a, **k: None)
    mod = _load_launcher()
    names = mod.load_models()
    # Exactly the two LLM entries (chat + legacy-no-type), sorted; nothing else.
    assert names == ["chat-llm", "legacy-no-type"], names
    for non_llm in ("an-embedding", "a-lora", "a-vae", "a-diffusion", "a-mmproj", "unclassified"):
        assert non_llm not in names


def test_is_llm_predicate():
    from localm.model_manager import is_llm, is_auto_chat_eligible
    assert is_llm({"model_type": "llm"}) is True
    assert is_llm({}) is True                       # legacy entry with no type -> llm
    assert is_llm({"model_type": "unknown"}) is False
    assert is_llm({"model_type": "lora"}) is False
    assert is_llm({"model_type": "vae"}) is False
    assert is_llm({"model_type": "text-encoder"}) is False
    assert is_llm("not a dict") is False
    # is_llm is STRICTER than is_auto_chat_eligible: a LoRA is auto-chat-eligible
    # (it is not 'unknown') yet is NOT an LLM, so the launcher must still hide it.
    assert is_auto_chat_eligible({"model_type": "lora"}) is True
    assert is_llm({"model_type": "lora"}) is False
