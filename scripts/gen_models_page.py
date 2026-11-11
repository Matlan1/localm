#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write docs/models.md, the list of every model kind localm runs, from localm's own tables.

    python scripts/gen_models_page.py           # rewrite docs/models.md (reads llama.cpp's
                                                # architecture list at the pinned tag online)
    python scripts/gen_models_page.py --check   # exit 1 when docs/models.md is out of date

``COVERED`` names every builtin plugin, inference backend and media runtime and the page
section that covers it; tests/test_models_page.py fails when one exists that is not named.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from localm import voice  # noqa: E402
from localm.inference import embedder  # noqa: E402
from localm.inference.backends import _hf_remote_compat, _hf_worker  # noqa: E402
from localm.inference.backends.llamacpp import _api, _drafting, mtmd_gen  # noqa: E402
from localm.media.koboldcpp import models as kobold  # noqa: E402
from localm.media.sdcpp import families  # noqa: E402
from localm.model_manager import gguf, registry, unsupported  # noqa: E402
from localm.plugins.builtin.exllama import plug as exllama  # noqa: E402
from localm.plugins.builtin.voice import audio_model  # noqa: E402
from localm.setup_llama import pins  # noqa: E402

PAGE = REPO / "docs" / "models.md"
ARCH_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/src/llama-arch.cpp"
_ARCH_RE = re.compile(r'\{\s*LLM_ARCH_[A-Z0-9_]+,\s*"([^"]+)"\s*\}')
_ARCH_LINE = re.compile(r"^<!-- llama\.cpp architectures at (\S+): (.*) -->$", re.M)

# Every builtin plugin, inference backend and media runtime, and where the page covers it
# (None: it runs no model of its own).
COVERED = {
    "plugins/builtin/chat": None, "plugins/builtin/coder": None, "plugins/builtin/jobs": None,
    "plugins/builtin/browser": None, "plugins/builtin/web": None, "plugins/builtin/mcp": None,
    "plugins/builtin/rag": "Embeddings and rerankers", "plugins/builtin/memory": "Embeddings and rerankers",
    "plugins/builtin/exllama": "Language models", "plugins/builtin/image": "Image",
    "plugins/builtin/video": "Video", "plugins/builtin/music": "Music",
    "plugins/builtin/voice": "Speech", "plugins/builtin/tts": "Speech",
    "inference/backends/llamacpp": "Language models", "inference/backends/gguf.py": "Language models",
    "inference/backends/hf.py": "Language models", "inference/backends/_hf_worker.py": "Language models",
    "media/sdcpp": "Image", "media/koboldcpp": "Music", "media/comfy_client.py": "Image",
    "media/comfy_shim": "Image", "inference/backends/base.py": None,
}

# Well-known families named in the summary line, each shown only while its architecture is listed.
EXAMPLES = (("Llama", "llama"), ("Qwen", "qwen3"), ("Gemma", "gemma4"), ("Mistral", "mistral3"),
            ("DeepSeek", "deepseek2"), ("GLM", "glm4moe"), ("Phi", "phi3"), ("gpt-oss", "gpt-oss"),
            ("Kimi", "kimi-k3"), ("MiniMax", "minimax-m2"), ("Nemotron", "nemotron_h"))

TYPE_TEXT = {
    "llm": "Language models: chat, completion, tools, and image or audio input with a projector.",
    "mmproj": "Projectors that give a language model image or audio input.",
    "embedding": "Embedding models for Knowledge (RAG) and memory, and rerankers.",
    "decision": "Decision models: answer a typed question with option probabilities "
                "(for example Perplexity pplx-decider). Registered, kept out of chat; no runner yet.",
    "tts": "Text to speech models.",
    "diffusion-unet": "Image and video generation checkpoints and diffusion models.",
    "text-encoder": "Text encoders used by image, video and music models.",
    "vae": "VAEs that turn image, video or music latents into output.",
    "lora": "LoRA adapters: GGUF adapters for language models, and LoRA files for image and video "
            "models.",
    "controlnet": "ControlNets that steer image generation from a reference image.",
    "textual-inversion": "Textual inversion embeddings for image prompts.",
    "upscaler": "Image upscalers (ESRGAN, Real-ESRGAN).",
    "segmentation": "Segmentation models (SAM, SAM 2). Registered; no runner yet.",
    "vocoder": "Vocoders that turn music latents into audio, from Hugging Face pipeline folders. "
               "Registered; no runner yet.",
    "planner": "Planner language models used by music generation (ACE-Step).",
}


def _names(items, sort=True) -> str:
    return ", ".join(f"`{x}`" for x in (sorted(items) if sort else items))


def fetch_arches(tag: str) -> list[str]:
    """The architecture names llama.cpp *tag* knows, read from its llama-arch.cpp."""
    with urllib.request.urlopen(ARCH_URL.format(tag=tag), timeout=30) as r:
        src = r.read().decode("utf-8")
    return sorted({a for a in _ARCH_RE.findall(src) if a not in ("clip", "(unknown)")})


def recorded_arches(text: str):
    """(tag, architectures) recorded in a generated page, or None."""
    m = _ARCH_LINE.search(text or "")
    return (m.group(1), m.group(2).split()) if m else None


def _kokoro() -> str:
    cfg = json.loads((REPO / "localm/plugins/builtin/tts/tts.example.json").read_text(encoding="utf-8"))
    return str(cfg.get("model", ""))


def render(arches: list[str], tag: str) -> str:
    non_chat = (set(gguf._GGUF_EMBEDDING_ARCHITECTURES) | set(gguf.GGUF_DIFFUSION_ARCHITECTURES)
                | set(gguf._GGUF_NON_CHAT_ARCHITECTURES) | set(gguf.GGUF_TEXT_ENCODER_ARCHITECTURES))
    decision = [a for a in arches if "decision" in a]
    chat = [a for a in arches if a not in non_chat and a not in decision]
    labels = families.FAMILY_LABELS
    img = sorted(families.IMAGE_FAMILIES, key=lambda f: labels.get(f, f))
    vid = sorted(families.VIDEO_FAMILIES, key=lambda f: labels.get(f, f))
    tts = [mtmd_gen.PIPELINE_NAMES[p] for p in sorted(mtmd_gen.SUPPORTED_PIPELINES)]
    tts_known = sorted(set(mtmd_gen.PIPELINE_NAMES.values()) - set(tts))
    native = families._GGUF_DIFFUSION_ARCH
    gguf_img = sorted(a for a, f in native.items() if f in families.IMAGE_FAMILIES)
    gguf_vid = sorted(a for a, f in native.items() if f in families.VIDEO_FAMILIES or f is None)
    gguf_comfy = sorted(set(gguf._GGUF_IMAGE_ARCHITECTURES) - set(native))
    draft_heads = sorted(a for a, why in gguf._GGUF_NON_CHAT_ARCHITECTURES.items()
                         if "draft head" in why)
    L = [
        "# Supported models",
        "",
        "<!-- Generated by scripts/gen_models_page.py from localm's own tables. Do not edit by hand. -->",
        f"<!-- llama.cpp architectures at {tag}: {' '.join(arches)} -->",
        "",
        "Everything localm can load or run, by what it does. Every list below is read from the "
        "tables localm itself uses to recognise a model, so this page and the program agree.",
        "",
        "## Formats",
        "",
        "- **GGUF**, also inside an Ollama store (`sha256-*` blobs), a llamafile, a Jan store, the "
        "Hugging Face cache or a split `-00001-of-0000N` set.",
        "- **Hugging Face folders** (`config.json` with `.safetensors`, `.bin`, `.pt` or `.pth` weights).",
        f"- **EXL2 and EXL3** folders: {_names(exllama.FORMATS.values())} (ExLlama, Linux with an NVIDIA GPU).",
        "- **Single checkpoint files** for image, video and music: `.gguf`, `.safetensors`, `.sft`, "
        "and `.pth` / `.pt` for upscalers.",
        "- **CTranslate2** Whisper models for speech to text, and **ONNX** for the browser voice.",
        "",
        "## Language models",
        "",
        f"- **GGUF through llama.cpp `{tag}`:** {len(chat)} architectures, among them "
        + ", ".join(name for name, arch in EXAMPLES if arch in chat) + ".",
        "",
        f"  <details><summary>All {len(chat)} GGUF language model architectures</summary>",
        "",
        f"  {_names(chat)}",
        "",
        "  </details>",
        "",
        "- **Image and audio input:** any GGUF chat model with its projector (`--mmproj`), "
        "and Hugging Face models whose processor takes images or audio.",
        "- **Hugging Face Transformers:** models whose `config.json` names an architecture ending in "
        f"{_names(_hf_worker._GENERATIVE_ARCH_SUFFIXES, sort=False)}, in full precision or "
        f"quantized ({len(unsupported._LOADER_QUANT_METHODS)} methods: "
        f"{_names(unsupported._LOADER_QUANT_METHODS)}). Models with bundled code load with "
        f"`hf_trust_remote_code` on; {_names(_hf_remote_compat._MODEL_TYPES)} is adapted to the "
        "bundled transformers.",
        f"- **EXL2 and EXL3:** through ExLlama ({_names(exllama.FORMATS.values())}).",
        f"- **Diffusion language models (GGUF):** {_names(gguf.GGUF_DIFFUSION_ARCHITECTURES)}.",
        f"- **Faster generation:** speculative decoding by {_names([s for s in _drafting.SPEC_SOURCES if s != 'off'], sort=False)}; "
        f"multi-token prediction on {_names(_api.MTP_GRAPH_ARCHITECTURES)}.",
        "- **LoRA adapters (GGUF)** attached to a GGUF model.",
        "",
        "## Embeddings and rerankers",
        "",
        f"- **GGUF embedding architectures:** {_names(gguf._GGUF_EMBEDDING_ARCHITECTURES)}, and any "
        "GGUF that declares a pooling type or is not causal.",
        f"- **Built-in embedding models:** {_names(embedder.KNOWN_EMBEDDING_MODELS)}.",
        "- **Hugging Face:** sentence-transformers folders and encoder models.",
        "- **Rerankers:** GGUF rerankers (rank pooling or a classifier head) and Hugging Face "
        "`*ForSequenceClassification` cross-encoders, Qwen3 rerankers included.",
        "",
        "## Speech",
        "",
        f"- **Speech to text:** Whisper ({_names(voice._STT_PINS, sort=False)}, or any "
        "faster-whisper repository), and any chat model that hears audio, for example "
        f"`{audio_model.SUGGESTED_PULL}` (Qwen3-ASR).",
        f"- **Text to speech on the server (GGUF):** {_names(tts)}, at `/v1/audio/speech`.",
        f"- **Text to speech in the browser:** Kokoro (`{_kokoro()}`).",
        "",
        "## Image",
        "",
        "- **Built-in (stable-diffusion.cpp):** "
        + ", ".join(f"{labels.get(f, f)} (`{f}`)" for f in img) + ".",
        "- **With them:** text encoders, VAEs, LoRA (LoRA, LoCon, LoHa, LoKr, OFT), "
        f"ControlNet and textual inversion on {_names(families.UNET_FAMILIES)}, reference "
        f"images on {_names(families.REFERENCE_FAMILIES)}, and ESRGAN / Real-ESRGAN upscalers.",
        f"- **GGUF image checkpoints, built in:** {_names(gguf_img)}.",
        f"- **GGUF text encoders:** {_names(gguf.GGUF_TEXT_ENCODER_ARCHITECTURES)} (T5, UMT5), "
        "for image and video models.",
        "- **ComfyUI:** any model a ComfyUI workflow loads, through the managed ComfyUI or your own, "
        f"including the GGUF architectures the built-in backend does not run: {_names(gguf_comfy)}.",
        "",
        "## Video",
        "",
        "- **Built-in (stable-diffusion.cpp):** "
        + ", ".join(f"{labels.get(f, f)} (`{f}`)" for f in vid) + ".",
        f"- **GGUF video checkpoints, built in:** {_names(gguf_vid)}.",
        "- **ComfyUI:** any video workflow (Wan 2.2 TI2V 5B ships as the default).",
        "",
        "## Music",
        "",
        f"- **Built-in (koboldcpp, ACE-Step 1.5 GGUF):** {_names(kobold.ARCHITECTURES.values())}, "
        f"from `{kobold.DEFAULT_REPO}` by default.",
        "- **ComfyUI:** ACE-Step v1 3.5B workflows.",
        "",
        "## Recognised but not run",
        "",
        f"- **Decision models** (GGUF, {_names(decision)} and any file with decision metadata) and "
        "**segmentation models (SAM)** are registered with their own type "
        "and kept out of chat; localm has no runner for them yet.",
        "- **Vocoders and Hugging Face pipeline-part folders** (diffusers-style transformer, VAE, "
        "vocoder or text-encoder folders) are typed but not run: the built-in music backend loads "
        "ACE-Step GGUF files and ComfyUI loads single checkpoint files.",
        "- **Hugging Face LoRA (PEFT) adapter folders** are typed `lora` but not attached to a model; "
        "language-model adapters attach as GGUF.",
        f"- **Speculative-decoding draft heads** ({_names(draft_heads)}) and "
        f"{_names(tts_known + ['wavtokenizer-dec'])} are recognised and refused for chat.",
        "",
        "## Model types",
        "",
        "Each registered model gets one of these types:",
        "",
    ]
    for t in sorted(registry.MODEL_TYPES - {"unknown"}):
        L.append(f"- **`{t}`:** {TYPE_TEXT[t]}")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    current = PAGE.read_text(encoding="utf-8") if PAGE.is_file() else ""
    if args.check:
        rec = recorded_arches(current)
        ok = rec is not None and rec[0] == pins._PINNED_TAG and current == render(rec[1], rec[0])
        if not ok:
            print("docs/models.md is out of date: run python scripts/gen_models_page.py")
            return 1
        return 0
    tag = pins._PINNED_TAG
    PAGE.write_text(render(fetch_arches(tag), tag), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
