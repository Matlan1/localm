# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic model type-detection, the 'unknown' sentinel, and the
lone-.safetensors parent-dir scan.

  * pull's HF classifier matches tags EXACTLY, never by substring (a tag that
    merely CONTAINS 'lora'/'vae' must not misclassify), and returns 'unknown' -
    not a silent 'llm' - when no hard signal resolves;
  * add_local records a deterministically-detected type, and 'unknown' (not 'llm')
    for an HF dir with no hard signal in config.json;
  * a lone .safetensors beside a config.json + tokenizer registers the DIRECTORY as
    an HF model; a lone .safetensors with no siblings is rejected with a PRECISE,
    actionable reason (never the bare "Not a model");
  * a type='unknown' model is runnable when named but is never auto-picked as the
    default chat model; its type is mutable via set_model_type / the CLI.
"""

import json
import os
import struct
import time
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager import gguf as _gguf_mod
from localm.model_manager.gguf import gguf_embedding_signal, gguf_is_mmproj
from localm.model_manager.pull import _hf_pipeline_tag_to_type


def _backdate(path, seconds=60):
    """Set path's mtime `seconds` in the past, so sync_models_dir's settle
    check (localm.model_manager.gguf._gguf_recently_written) reads it as
    settled rather than possibly still mid-copy - these tests are about type
    detection, not the settle window, and would otherwise flake on the
    write-then-immediately-sync timing."""
    old = time.time() - seconds
    os.utime(path, (old, old))


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Throwaway data dir with the config + model_manager path singletons redirected
    (mirrors tests/test_add_local_folder_of_ggufs.isolated_home): the real add_local
    + load_registry round-trip through registry.json under tmp_path."""
    import localm.config as cfg
    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")
    return home


def _hf_dir(root, name, *, architectures=None, adapter=False):
    """A minimal HuggingFace model directory (config.json + tokenizer + a weight)."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    conf = {}
    if architectures is not None:
        conf["architectures"] = architectures
    (d / "config.json").write_text(json.dumps(conf), encoding="utf-8")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00\x01\x02" + name.encode())
    if adapter:
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
#  pull.py HF classifier: exact tag match + unknown sentinel                   #
# --------------------------------------------------------------------------- #

def _patch_hf(monkeypatch, payload):
    import localm.discover as discover
    monkeypatch.setattr(discover, "_get", lambda url, params=None: payload)


def test_hf_tag_substring_does_not_misclassify(monkeypatch):
    # 'exploration' CONTAINS 'lora' as a substring (...p-LORA-tion...). Exact
    # matching must NOT call this repo a LoRA.
    _patch_hf(monkeypatch, {"pipeline_tag": "text-generation", "tags": ["exploration"]})
    assert _hf_pipeline_tag_to_type("owner/repo") == "llm"


def test_hf_vae_substring_does_not_misclassify(monkeypatch):
    # A tag merely CONTAINING 'vae' as a substring must not be read as a VAE.
    _patch_hf(monkeypatch, {"pipeline_tag": None, "tags": ["vae-experiments"]})
    assert _hf_pipeline_tag_to_type("owner/repo") == "unknown"


def test_hf_exact_tag_still_classifies(monkeypatch):
    # Exact tag tokens still classify (positive control - a real VAE tag).
    _patch_hf(monkeypatch, {"pipeline_tag": None, "tags": ["vae"]})
    assert _hf_pipeline_tag_to_type("owner/repo") == "vae"


def test_hf_unresolved_is_unknown_not_llm(monkeypatch):
    # No pipeline_tag, no recognised tag/library -> 'unknown', not a silent 'llm'.
    _patch_hf(monkeypatch, {"pipeline_tag": None, "tags": ["some-random-tag"]})
    assert _hf_pipeline_tag_to_type("owner/repo") == "unknown"


def test_hf_query_failure_is_unknown(monkeypatch):
    import localm.discover as discover

    def _boom(url, params=None):
        raise RuntimeError("offline")

    monkeypatch.setattr(discover, "_get", _boom)
    assert _hf_pipeline_tag_to_type("owner/repo") == "unknown"


def test_hf_text_generation_is_llm(monkeypatch):
    _patch_hf(monkeypatch, {"pipeline_tag": "text-generation", "tags": []})
    assert _hf_pipeline_tag_to_type("owner/repo") == "llm"


def test_hf_image_is_diffusion(monkeypatch):
    _patch_hf(monkeypatch, {"pipeline_tag": "text-to-image", "tags": []})
    assert _hf_pipeline_tag_to_type("owner/repo") == "diffusion-unet"


def test_hf_diffusion_lora_precedence(monkeypatch):
    # XLabs-AI/flux-RealismLora: pipeline_tag=text-to-image (inherited from its
    # FLUX base model) AND an exact 'lora' tag on the same repo. The tag wins -
    # classify_hf_metadata checks it before the diffusion pipeline_tag branch.
    _patch_hf(monkeypatch, {
        "pipeline_tag": "text-to-image", "library_name": "diffusers",
        "tags": ["diffusers", "lora", "Stable Diffusion", "image-generation",
                 "Flux", "text-to-image",
                 "base_model:adapter:black-forest-labs/FLUX.1-dev"],
    })
    assert _hf_pipeline_tag_to_type("XLabs-AI/flux-RealismLora") == "lora"


# --------------------------------------------------------------------------- #
#  add_local: lone .safetensors dir-scan + HF-dir detection                    #
# --------------------------------------------------------------------------- #

def test_lone_safetensors_beside_config_registers_directory(tmp_path, isolated_home):
    # A .safetensors is not itself loadable; beside a config.json + tokenizer it
    # is part of an HF model dir -> register the DIRECTORY.
    d = _hf_dir(tmp_path, "my-llm", architectures=["LlamaForCausalLM"])
    weight = d / "model.safetensors"
    assert mm.add_local(str(weight)) is True
    reg = mm.load_registry()
    assert "my-llm" in reg
    assert Path(reg["my-llm"]["path"]).is_dir()      # the dir, not the bare file
    assert reg["my-llm"]["model_type"] == "llm"      # detected from architectures


def test_lone_safetensors_without_config_precise_reject(tmp_path, isolated_home, monkeypatch):
    # A lone .safetensors with no config.json sibling: reject with a PRECISE reason,
    # never the bare "Not a model".
    from localm.model_manager import _shared
    printed = []
    monkeypatch.setattr(_shared.console, "print",
                        lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    d = tmp_path / "loose"
    d.mkdir()
    weight = d / "weights.safetensors"
    weight.write_bytes(b"\x00\x01")
    assert mm.add_local(str(weight)) is False
    assert mm.load_registry() == {}
    out = "\n".join(printed)
    assert "Not a model" not in out          # not the generic rejection
    assert "config.json" in out              # the precise, actionable reason


def test_add_local_hf_dir_causal_arch_is_llm(tmp_path, isolated_home):
    d = _hf_dir(tmp_path, "chatty", architectures=["Qwen2ForCausalLM"])
    assert mm.add_local(str(d)) is True
    assert mm.load_registry()["chatty"]["model_type"] == "llm"


def test_add_local_hf_dir_adapter_is_lora(tmp_path, isolated_home):
    d = _hf_dir(tmp_path, "an-adapter", architectures=None, adapter=True)
    assert mm.add_local(str(d)) is True
    assert mm.load_registry()["an-adapter"]["model_type"] == "lora"


def test_add_local_hf_dir_no_arch_is_unknown(tmp_path, isolated_home):
    # No hard signal in config.json -> 'unknown', never a silent 'llm'.
    d = _hf_dir(tmp_path, "mystery", architectures=None)
    assert mm.add_local(str(d)) is True
    assert mm.load_registry()["mystery"]["model_type"] == "unknown"


def test_add_local_gguf_still_llm(tmp_path, isolated_home):
    # Guard: a bare .gguf is a llama.cpp text model -> stays 'llm'.
    f = tmp_path / "plain.gguf"
    f.write_bytes(b"GGUF\x00\x00\x00\x00plain")
    assert mm.add_local(str(f)) is True
    assert mm.load_registry()["plain"]["model_type"] == "llm"


# --------------------------------------------------------------------------- #
#  unknown is runnable-by-name but never auto-picked as the chat default       #
# --------------------------------------------------------------------------- #

def test_is_auto_chat_eligible():
    from localm.model_manager import is_auto_chat_eligible
    assert is_auto_chat_eligible({"model_type": "llm"}) is True
    assert is_auto_chat_eligible({}) is True                       # legacy entry = llm
    assert is_auto_chat_eligible({"model_type": "unknown"}) is False
    # An 'embedding' entry must never be auto-picked as the default CHAT model:
    # it loads via a dedicated embeddings-mode context, not the causal chat path,
    # and setup-embeddings can register one into the main registry.
    assert is_auto_chat_eligible({"model_type": "embedding"}) is False


def test_unknown_not_auto_selected_but_runnable_by_name(monkeypatch):
    from localm.plugins.mcpserver.server import EngineCache
    reg = {"zeta-unknown": {"path": "x", "model_type": "unknown"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    ec = EngineCache(default_model=None)
    # no name -> must NOT auto-pick the unknown model
    with pytest.raises(ValueError):
        ec.resolve_model(None)
    # explicit name still resolves (runnable when named)
    assert ec.resolve_model("zeta-unknown") == "zeta-unknown"


def test_embedding_not_auto_selected_but_runnable_by_name(monkeypatch):
    # Same guarantee as the 'unknown' case above, for 'embedding': an
    # embedding-only registry must not auto-load one of them as the chat model,
    # but it stays runnable when named explicitly.
    from localm.plugins.mcpserver.server import EngineCache
    reg = {"zeta-embed": {"path": "x", "model_type": "embedding"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    ec = EngineCache(default_model=None)
    with pytest.raises(ValueError):
        ec.resolve_model(None)
    assert ec.resolve_model("zeta-embed") == "zeta-embed"


def test_autopick_prefers_llm_over_unknown(monkeypatch):
    from localm.plugins.mcpserver.server import EngineCache
    reg = {"a-unknown": {"path": "x", "model_type": "unknown"},
           "b-llm": {"path": "y", "model_type": "llm"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    assert EngineCache(default_model=None).resolve_model(None) == "b-llm"


def test_autopick_prefers_llm_over_embedding(monkeypatch):
    from localm.plugins.mcpserver.server import EngineCache
    reg = {"a-embed": {"path": "x", "model_type": "embedding"},
           "b-llm": {"path": "y", "model_type": "llm"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    assert EngineCache(default_model=None).resolve_model(None) == "b-llm"


# --------------------------------------------------------------------------- #
#  Type is mutable (set_model_type helper + CLI)                               #
# --------------------------------------------------------------------------- #

def test_set_model_type_changes_and_validates(tmp_path, isolated_home):
    from localm.model_manager import set_model_type
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF\x00\x00\x00\x00m")
    assert mm.add_local(str(f)) is True
    assert mm.load_registry()["m"]["model_type"] == "llm"
    assert set_model_type("m", "unknown") is True
    assert mm.load_registry()["m"]["model_type"] == "unknown"
    # still runnable by name after retyping
    assert mm.get_model_info("m") is not None
    # rejects a bogus type and a missing model
    assert set_model_type("m", "not-a-type") is False
    assert set_model_type("does-not-exist", "llm") is False


def test_cli_set_type_command(tmp_path, monkeypatch):
    import localm.config as cfg
    from click.testing import CliRunner
    import localm.cli as cli_pkg

    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    for attr, val in (("HOME_DIR", home), ("MODELS_DIR", home / "models"),
                      ("CONFIG_FILE", home / "config.json"),
                      ("REGISTRY_FILE", home / "registry.json")):
        monkeypatch.setattr(cfg, attr, val)
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")

    f = home / "models" / "cli_model.gguf"
    f.write_bytes(b"GGUF\x00\x00\x00\x00c")
    assert mm.add_local(str(f)) is True

    runner = CliRunner()
    res = runner.invoke(cli_pkg.main, ["set-type", "cli_model", "vae"])
    assert res.exit_code == 0, res.output
    assert mm.load_registry()["cli_model"]["model_type"] == "vae"
    # an out-of-vocab type is rejected by the click.Choice constraint
    res2 = runner.invoke(cli_pkg.main, ["set-type", "cli_model", "bogus"])
    assert res2.exit_code != 0


# --------------------------------------------------------------------------- #
#  A distinct 'embedding' MODEL_TYPES value, detected from HARD GGUF metadata  #
#  (general.architecture / *.pooling_type), never a filename guess. Covers     #
#  gguf.py (gguf_embedding_signal), registry.py (_detect_local_model_type +    #
#  sync_models_dir), pull.py (the HF pipeline_tag classifier + the             #
#  post-download auto-upgrade) and cli/models.py (--type).                     #
# --------------------------------------------------------------------------- #

def _build_gguf_bytes(architecture: str, extra_kv: "dict | None" = None) -> bytes:
    """Construct a minimal-but-structurally-valid GGUF v3 file byte-for-byte in
    the layout ``_gguf_metadata_probe`` (localm/model_manager/gguf.py) parses:
    magic ``b"GGUF"`` + uint32 version + uint64 tensor_count + uint64 kv_count,
    followed by exactly ``kv_count`` length-prefixed (key, type, value) triples.
    Every KV here is written as GGUF_TYPE_STRING(8) - the real format's own
    string encoding is a uint64 length prefix then the raw utf-8 bytes, used for
    both the key and the value. ``extra_kv`` entries only need to exist for
    ``gguf_embedding_signal``'s pooling_type check, which keys off the KEY name
    alone, so a placeholder string value is fine. Padded with trailing zero
    bytes (never parsed - the loop stops after ``kv_count`` entries) so the file
    clears gguf.py's own ``_GGUF_MIN_BYTES`` floor, exactly like a real model.
    """
    extra_kv = extra_kv or {}

    def _kv_string(key: str, value: str) -> bytes:
        kb = key.encode("utf-8")
        vb = value.encode("utf-8")
        return (
            struct.pack("<Q", len(kb)) + kb
            + struct.pack("<I", 8)          # GGUF_TYPE_STRING
            + struct.pack("<Q", len(vb)) + vb
        )

    kv_count = 1 + len(extra_kv)
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)             # version 3
    buf += struct.pack("<Q", 0)             # tensor_count
    buf += struct.pack("<Q", kv_count)      # kv_count
    buf += _kv_string("general.architecture", architecture)
    for key, value in extra_kv.items():
        buf += _kv_string(key, str(value))

    floor = _gguf_mod._GGUF_MIN_BYTES
    if len(buf) < floor:
        buf += b"\x00" * (floor - len(buf))
    return bytes(buf)


# ------------------------- gguf_embedding_signal ------------------------- #

def test_gguf_embedding_signal_bert_architecture_true(tmp_path):
    f = tmp_path / "bert-embed.gguf"
    f.write_bytes(_build_gguf_bytes("bert"))
    assert gguf_embedding_signal(f) is True


def test_gguf_embedding_signal_llama_no_pooling_false(tmp_path):
    # A causal-chat architecture with no pooling_type key at all -> not an
    # embedding model.
    f = tmp_path / "llama-chat.gguf"
    f.write_bytes(_build_gguf_bytes("llama"))
    assert gguf_embedding_signal(f) is False


def test_gguf_embedding_signal_qwen3_pooling_type_true(tmp_path):
    # The decoder-reuse case: general.architecture stays qwen3 (identical to the
    # chat variant), but a pooling-configured export carries a
    # <arch>.pooling_type key the chat variant never writes - that key alone is
    # the signal.
    f = tmp_path / "qwen3-embed.gguf"
    f.write_bytes(_build_gguf_bytes("qwen3", {"qwen3.pooling_type": "1"}))
    assert gguf_embedding_signal(f) is True


def test_gguf_embedding_signal_truncated_file_no_crash(tmp_path):
    # GGUF magic present but the header is cut off before tensor/kv counts can
    # even be read - must degrade to "no signal", never raise.
    f = tmp_path / "truncated.gguf"
    f.write_bytes(b"GGUF" + b"\x03\x00\x00")
    assert gguf_embedding_signal(f) is False


def test_gguf_embedding_signal_bert_survives_huge_vocab_truncation(tmp_path):
    # general.architecture=bert resolves EARLY in the KV walk, but a real
    # embedding model's tokenizer vocab array (multilingual models like bge-m3
    # carry 250k+ tokens) can push the metadata block past the bounded probe
    # read. The KV walk breaks out as soon as EITHER signal is definitively known
    # and preserves whatever was already resolved if a later key trips a parse
    # error, so this returns True regardless of what follows.
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)              # version 3
    buf += struct.pack("<Q", 0)               # tensor_count
    buf += struct.pack("<Q", 2)               # kv_count: architecture + huge vocab array

    def _kv_string(key: str, value: str) -> bytes:
        kb, vb = key.encode("utf-8"), value.encode("utf-8")
        return (struct.pack("<Q", len(kb)) + kb
                + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb)

    buf += _kv_string("general.architecture", "bert")

    # A huge tokenizer.ggml.tokens array: ARRAY(9) of STRING(8) elements, many
    # more than fit in the probe bound - simulates a real multilingual vocab.
    key = b"tokenizer.ggml.tokens"
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", 9)               # GGUF_TYPE_ARRAY
    n_tokens = 400_000
    buf += struct.pack("<I", 8)               # element type: STRING
    buf += struct.pack("<Q", n_tokens)
    token = b"tok"
    for _ in range(n_tokens):
        buf += struct.pack("<Q", len(token)) + token   # ~5 MB total, past the 4 MB probe bound

    f = tmp_path / "bert-huge-vocab.gguf"
    f.write_bytes(bytes(buf))
    assert gguf_embedding_signal(f) is True


def test_gguf_embedding_signal_non_gguf_file_no_crash(tmp_path):
    f = tmp_path / "not-a-model.bin"
    f.write_bytes(b"this is definitely not a gguf file" * 50)
    assert gguf_embedding_signal(f) is False


# ------------------------------ add_local --------------------------------- #

def test_add_local_gguf_bert_architecture_is_embedding(tmp_path, isolated_home):
    f = tmp_path / "my-embedder.gguf"
    f.write_bytes(_build_gguf_bytes("bert"))
    assert mm.add_local(str(f)) is True
    assert mm.load_registry()["my-embedder"]["model_type"] == "embedding"


def test_add_local_full_gguf_llama_architecture_still_llm(tmp_path, isolated_home):
    # A fully valid GGUF whose metadata DOES parse (a real architecture, no
    # pooling key), driving the negative path through the real parser rather than
    # the parse-failed fallback.
    f = tmp_path / "my-chat-model.gguf"
    f.write_bytes(_build_gguf_bytes("llama"))
    assert mm.add_local(str(f)) is True
    assert mm.load_registry()["my-chat-model"]["model_type"] == "llm"


# --------------------------- sync_models_dir ------------------------------- #

def test_sync_models_dir_discovers_embedding_gguf(isolated_home):
    dest = isolated_home / "models" / "auto-embed.gguf"
    dest.write_bytes(_build_gguf_bytes("nomic-bert"))
    _backdate(dest)
    result = mm.sync_models_dir()
    assert result.added == 1
    assert mm.load_registry()["auto-embed"]["model_type"] == "embedding"


# --------------------------- _hf_pipeline_tag_to_type ---------------------- #

def test_hf_feature_extraction_is_embedding(monkeypatch):
    _patch_hf(monkeypatch, {"pipeline_tag": "feature-extraction", "tags": []})
    assert _hf_pipeline_tag_to_type("owner/repo") == "embedding"


def test_hf_sentence_similarity_is_embedding(monkeypatch):
    _patch_hf(monkeypatch, {"pipeline_tag": "sentence-similarity", "tags": []})
    assert _hf_pipeline_tag_to_type("owner/repo") == "embedding"


# ------------------------------ pull_model --------------------------------- #

def test_pull_model_auto_upgrades_gguf_to_embedding(tmp_path, isolated_home, monkeypatch):
    # pull_model's default model_type="auto" resolves a bare *.gguf spec to
    # 'llm' up front (the container format is a hard signal on its own), but
    # _pull_gguf_file then probes the freshly-downloaded file's OWN metadata and
    # upgrades an auto-resolved 'llm' to 'embedding' when the bytes say so.
    import huggingface_hub
    import requests

    embedding_bytes = _build_gguf_bytes("bert")

    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.write_bytes(embedding_bytes)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    # Avoid any real network: no HF metadata digest, and the disk-space
    # preflight HEAD fails closed to total_size=0 (still a benign no-op path).
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
    monkeypatch.setattr(
        requests, "head",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no network in tests")))

    assert mm.pull_model("owner/repo:auto-embed.gguf") is True
    assert mm.load_registry()["auto-embed"]["model_type"] == "embedding"


def test_pull_model_explicit_llm_type_not_overridden(tmp_path, isolated_home, monkeypatch):
    # Companion negative case: the SAME embedding-signal bytes pulled with an
    # explicit model_type="llm" must stay 'llm' - the auto-upgrade in
    # _pull_gguf_file is gated on type_is_auto and must never override an
    # explicitly-passed --type.
    import huggingface_hub
    import requests

    embedding_bytes = _build_gguf_bytes("bert")

    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.write_bytes(embedding_bytes)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
    monkeypatch.setattr(
        requests, "head",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no network in tests")))

    assert mm.pull_model("owner/repo:forced-llm.gguf", model_type="llm") is True
    assert mm.load_registry()["forced-llm"]["model_type"] == "llm"


# --------------------------------------------------------------------------- #
#  A distinct 'mmproj' MODEL_TYPES value, detected from HARD GGUF metadata     #
#  (general.architecture == clip), never a filename guess. Covers gguf.py      #
#  (gguf_is_mmproj), registry.py (_detect_local_model_type + sync_models_dir)  #
#  and pull.py (the post-download auto-upgrade). `localm pull ... --mmproj`    #
#  leaves the projector unregistered on disk, and sync_models_dir must not     #
#  pick it back up as a plain 'llm'.                                           #
# --------------------------------------------------------------------------- #

# ---------------------------- gguf_is_mmproj ------------------------------- #

def test_gguf_is_mmproj_clip_architecture_true(tmp_path):
    f = tmp_path / "mmproj-model.gguf"
    f.write_bytes(_build_gguf_bytes("clip"))
    assert gguf_is_mmproj(f) is True


def test_gguf_is_mmproj_llama_architecture_false(tmp_path):
    # A causal-chat architecture must never be flagged as a projector.
    f = tmp_path / "llama-chat.gguf"
    f.write_bytes(_build_gguf_bytes("llama"))
    assert gguf_is_mmproj(f) is False


def test_gguf_is_mmproj_embedding_architecture_false(tmp_path):
    # Cross-contamination guard: an embedding architecture must not also read
    # as a vision projector - the two are independent hard-metadata checks
    # over the same probe, so nothing accidentally OR's them together.
    f = tmp_path / "bert-embed.gguf"
    f.write_bytes(_build_gguf_bytes("bert"))
    assert gguf_is_mmproj(f) is False


def test_gguf_is_mmproj_truncated_file_no_crash(tmp_path):
    f = tmp_path / "truncated.gguf"
    f.write_bytes(b"GGUF" + b"\x03\x00\x00")
    assert gguf_is_mmproj(f) is False


def test_gguf_is_mmproj_non_gguf_file_no_crash(tmp_path):
    f = tmp_path / "not-a-model.bin"
    f.write_bytes(b"this is definitely not a gguf file" * 50)
    assert gguf_is_mmproj(f) is False


# ------------------------------ add_local --------------------------------- #

def test_add_local_gguf_clip_architecture_is_mmproj(tmp_path, isolated_home):
    f = tmp_path / "my-mmproj.gguf"
    f.write_bytes(_build_gguf_bytes("clip"))
    assert mm.add_local(str(f)) is True
    assert mm.load_registry()["my-mmproj"]["model_type"] == "mmproj"


# --------------------------- sync_models_dir ------------------------------- #

def test_sync_models_dir_discovers_mmproj_gguf(isolated_home):
    dest = isolated_home / "models" / "mmproj-gemma-3-4b-it-f16.gguf"
    dest.write_bytes(_build_gguf_bytes("clip"))
    _backdate(dest)
    result = mm.sync_models_dir()
    assert result.added == 1
    assert mm.load_registry()["mmproj-gemma-3-4b-it-f16"]["model_type"] == "mmproj"


# --------------------------------- pull_model ------------------------------ #

def test_pull_model_auto_upgrades_gguf_to_mmproj(tmp_path, isolated_home, monkeypatch):
    # pull_model's default model_type=auto resolves a bare *.gguf spec to 'llm'
    # up front, and _pull_gguf_file then probes the freshly-downloaded file's OWN
    # metadata and upgrades an auto-resolved 'llm' to 'mmproj' when the bytes say
    # so, reached here by a direct pull of a projector file.
    import huggingface_hub
    import requests

    mmproj_bytes = _build_gguf_bytes("clip")

    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.write_bytes(mmproj_bytes)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
    monkeypatch.setattr(
        requests, "head",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no network in tests")))

    assert mm.pull_model("owner/repo:auto-mmproj.gguf") is True
    assert mm.load_registry()["auto-mmproj"]["model_type"] == "mmproj"


def test_pull_model_explicit_llm_type_not_overridden_for_mmproj_bytes(
        tmp_path, isolated_home, monkeypatch):
    # Companion negative case: the SAME clip-architecture bytes pulled with an
    # explicit model_type="llm" must stay 'llm' - the auto-upgrade is gated on
    # type_is_auto and must never override an explicitly-passed --type.
    import huggingface_hub
    import requests

    mmproj_bytes = _build_gguf_bytes("clip")

    def _fake_download(repo_id, filename, local_dir, **kw):
        p = Path(local_dir) / filename
        p.write_bytes(mmproj_bytes)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(mm, "_hf_file_sha256", lambda repo_id, filename: None)
    monkeypatch.setattr(
        requests, "head",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no network in tests")))

    assert mm.pull_model("owner/repo:forced-llm-mmproj.gguf", model_type="llm") is True
    assert mm.load_registry()["forced-llm-mmproj"]["model_type"] == "llm"


# ---------------------------------- CLI ------------------------------------ #

def test_cli_add_explicit_type_overrides_detection(tmp_path, monkeypatch):
    import localm.config as cfg
    from click.testing import CliRunner
    import localm.cli as cli_pkg

    home = tmp_path / ".localm"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    for attr, val in (("HOME_DIR", home), ("MODELS_DIR", home / "models"),
                      ("CONFIG_FILE", home / "config.json"),
                      ("REGISTRY_FILE", home / "registry.json")):
        monkeypatch.setattr(cfg, attr, val)
    monkeypatch.setattr(mm, "MODELS_DIR", home / "models")
    monkeypatch.setattr(mm, "REGISTRY_FILE", home / "registry.json")

    # A llama-architecture GGUF with no pooling key would auto-detect as 'llm' -
    # the explicit --type embedding below must win regardless (explicit choice
    # is never silently overridden by detection).
    f = tmp_path / "chat-model.gguf"
    f.write_bytes(_build_gguf_bytes("llama"))

    runner = CliRunner()
    res = runner.invoke(cli_pkg.main, ["add", str(f), "--type", "embedding"])
    assert res.exit_code == 0, res.output
    assert mm.load_registry()["chat-model"]["model_type"] == "embedding"

    # An out-of-vocabulary --type is rejected by click's Choice constraint.
    res2 = runner.invoke(cli_pkg.main, ["add", str(f), "--type", "bogus-not-a-type"])
    assert res2.exit_code != 0


# --------------------------- setup-embeddings ------------------------------ #

def test_setup_embeddings_registers_once_not_twice(isolated_home, monkeypatch):
    from click.testing import CliRunner
    from localm.cli.maintenance import setup_embeddings
    from localm.inference import embedder

    emb_dir = isolated_home / "models" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    fake = emb_dir / "bge-small-en-v1.5-q4_k_m.gguf"
    fake.write_bytes(_build_gguf_bytes("bert"))

    # Stub the DOWNLOADER (the environment), not the sync logic under test; the
    # fake file sits under <home>/models/embeddings so the registry-sync branch
    # fires.
    monkeypatch.setattr(embedder, "resolve_embedding_model_path",
                        lambda allow_download=True: str(fake))

    runner = CliRunner()
    res1 = runner.invoke(setup_embeddings, [])
    assert res1.exit_code == 0, res1.output
    reg1 = mm.load_registry()
    assert len(reg1) == 1
    name1 = next(iter(reg1))
    assert reg1[name1]["model_type"] == "embedding"

    # A second invocation resolves to the SAME path again - must not create a
    # duplicate registry entry (find_aliases_by_path guards it).
    res2 = runner.invoke(setup_embeddings, [])
    assert res2.exit_code == 0, res2.output
    reg2 = mm.load_registry()
    assert len(reg2) == len(reg1)
    assert set(reg2) == set(reg1)


# --------------- Architecture and expert count on the registry ------------- #
# A local model's registry entry carries its own header's architecture and MoE
# expert_count, captured once at registration. expert_count=0 is a CONFIRMED
# fact (the header was read and genuinely has no experts) and stays written and
# distinct from an entry that was never checked at all (the key absent
# entirely).

def _build_gguf_bytes_with_expert_count(architecture: str, expert_count: int) -> bytes:
    """Like _build_gguf_bytes, but writes a REAL uint32 '<architecture>.expert_count'
    key (GGUF type 4) instead of a placeholder string - gguf_expert_count's own
    reader (_gguf_read_scalar) only accepts a fixed-width numeric type, so a
    string-typed placeholder (as the base helper writes for every OTHER key)
    would silently fail to parse and always read back as 0/unknown, hiding a
    real bug in this exact test."""
    def _kv_string(key: str, value: str) -> bytes:
        kb = key.encode("utf-8")
        vb = value.encode("utf-8")
        return (struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8)
                + struct.pack("<Q", len(vb)) + vb)

    def _kv_uint32(key: str, value: int) -> bytes:
        kb = key.encode("utf-8")
        return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)

    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)
    buf += struct.pack("<Q", 2)   # kv_count: architecture + expert_count
    buf += _kv_string("general.architecture", architecture)
    buf += _kv_uint32(f"{architecture}.expert_count", expert_count)

    floor = _gguf_mod._GGUF_MIN_BYTES
    if len(buf) < floor:
        buf += b"\x00" * (floor - len(buf))
    return bytes(buf)


class TestGgufRegistryMetadata:
    def test_confirmed_moe_persists_architecture_and_expert_count(self, tmp_path):
        f = tmp_path / "moe.gguf"
        f.write_bytes(_build_gguf_bytes_with_expert_count("qwen3moe", 8))
        assert _gguf_mod.gguf_registry_metadata(f) == {"architecture": "qwen3moe", "expert_count": 8}

    def test_confirmed_dense_persists_a_real_zero_not_none(self, tmp_path):
        f = tmp_path / "dense.gguf"
        f.write_bytes(_build_gguf_bytes("llama"))   # no expert_count key at all
        result = _gguf_mod.gguf_registry_metadata(f)
        assert result["architecture"] == "llama"
        assert result["expert_count"] == 0, "a confirmed-read dense model must store 0, not None"

    def test_unreadable_file_reports_unknown_not_a_false_zero(self, tmp_path):
        f = tmp_path / "corrupt.gguf"
        f.write_bytes(b"NOT A REAL GGUF HEADER AT ALL" + b"\x00" * 1024)
        result = _gguf_mod.gguf_registry_metadata(f)
        assert result == {"architecture": None, "expert_count": None}, \
            "an unparseable file must report unknown (None), never a false 0/confirmed-dense"

    def test_meta_param_avoids_a_second_probe_read(self, tmp_path, monkeypatch):
        f = tmp_path / "moe.gguf"
        f.write_bytes(_build_gguf_bytes_with_expert_count("qwen3moe", 4))
        calls = []
        real_probe = _gguf_mod._gguf_metadata_probe
        monkeypatch.setattr(_gguf_mod, "_gguf_metadata_probe",
                            lambda p: (calls.append(1), real_probe(p))[1])
        meta = _gguf_mod._gguf_metadata_probe(f)
        calls.clear()
        _gguf_mod.gguf_registry_metadata(f, meta=meta)
        assert calls == [], "passing a pre-computed meta must not trigger a second probe read"


class TestAddLocalPersistsArchAndExpertCount:
    def test_add_local_persists_moe_metadata(self, tmp_path, isolated_home):
        f = tmp_path / "big-moe.gguf"
        f.write_bytes(_build_gguf_bytes_with_expert_count("qwen3moe", 8))
        assert mm.add_local(str(f)) is True
        entry = mm.load_registry()["big-moe"]
        assert entry["architecture"] == "qwen3moe"
        assert entry["expert_count"] == 8

    def test_add_local_persists_confirmed_dense_as_zero(self, tmp_path, isolated_home):
        f = tmp_path / "plain-llama.gguf"
        f.write_bytes(_build_gguf_bytes("llama"))
        assert mm.add_local(str(f)) is True
        entry = mm.load_registry()["plain-llama"]
        assert entry["architecture"] == "llama"
        assert entry["expert_count"] == 0

    def test_add_local_with_explicit_type_override_still_captures_metadata(self, tmp_path, isolated_home):
        # An explicit --type must not skip reading the file's own header - the
        # type LABEL and the architecture/expert_count FACTS are orthogonal.
        f = tmp_path / "forced.gguf"
        f.write_bytes(_build_gguf_bytes_with_expert_count("qwen3moe", 8))
        assert mm.add_local(str(f), model_type="embedding") is True
        entry = mm.load_registry()["forced"]
        assert entry["model_type"] == "embedding"
        assert entry["architecture"] == "qwen3moe"
        assert entry["expert_count"] == 8

    def test_hf_dir_entry_has_no_gguf_metadata(self, tmp_path, isolated_home):
        d = _hf_dir(tmp_path, "an-hf-model", architectures=["LlamaForCausalLM"])
        assert mm.add_local(str(d)) is True
        entry = mm.load_registry()["an-hf-model"]
        assert "architecture" not in entry, "an HF dir has no GGUF header to read"
        assert "expert_count" not in entry


class TestSyncModelsDirBackfillsArchAndExpertCount:
    def test_sync_backfills_a_pre_existing_entry_missing_the_fields(self, isolated_home):
        # A registry entry lacking architecture/expert_count, written directly
        # via _register with neither kwarg.
        dest = isolated_home / "models" / "legacy-moe.gguf"
        dest.write_bytes(_build_gguf_bytes_with_expert_count("qwen3moe", 8))
        mm._register("legacy-moe", dest, "local", model_type="llm")
        before = mm.load_registry()["legacy-moe"]
        assert "architecture" not in before and "expert_count" not in before

        result = mm.sync_models_dir()
        assert result.backfilled == 1
        assert result.changed is True

        after = mm.load_registry()["legacy-moe"]
        assert after["architecture"] == "qwen3moe"
        assert after["expert_count"] == 8

    def test_sync_never_overwrites_an_already_resolved_entry(self, isolated_home):
        # Even a stored 0/falsy value must never be re-read or clobbered - the
        # presence of the KEY (not its truthiness) is what "already resolved"
        # means throughout this feature.
        dest = isolated_home / "models" / "already-known.gguf"
        dest.write_bytes(_build_gguf_bytes_with_expert_count("qwen3moe", 99))
        mm._register("already-known", dest, "local", model_type="llm",
                     architecture="something-else", expert_count=0)

        result = mm.sync_models_dir()
        assert result.backfilled == 0

        after = mm.load_registry()["already-known"]
        assert after["architecture"] == "something-else", "an existing value must never be re-read"
        assert after["expert_count"] == 0

    def test_sync_backfill_is_capped_per_call(self, isolated_home):
        # _BACKFILL_CAP = 5: a user with many pre-existing models must not pay
        # for reading every one of them on a single sync call.
        for i in range(8):
            dest = isolated_home / "models" / f"legacy-{i}.gguf"
            dest.write_bytes(_build_gguf_bytes("llama"))
            mm._register(f"legacy-{i}", dest, "local", model_type="llm")

        result = mm.sync_models_dir()
        assert result.backfilled == 5, "the cap must actually bound the work done in one call"

        reg = mm.load_registry()
        backfilled_now = sum(1 for n in reg if "architecture" in reg[n])
        assert backfilled_now == 5

        # A second call makes further progress on the remaining entries.
        result2 = mm.sync_models_dir()
        assert result2.backfilled == 3
        reg2 = mm.load_registry()
        assert all("architecture" in reg2[n] for n in reg2)

    def test_sync_does_not_backfill_a_non_gguf_entry(self, isolated_home):
        d = _hf_dir(isolated_home, "an-hf-model", architectures=["LlamaForCausalLM"])
        mm._register("an-hf-model", d, "local", model_type="llm")
        result = mm.sync_models_dir()
        assert result.backfilled == 0
        assert "architecture" not in mm.load_registry()["an-hf-model"]
