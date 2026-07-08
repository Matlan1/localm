# SPDX-License-Identifier: AGPL-3.0-or-later
"""Branch A: deterministic model type-detection, the 'unknown' sentinel, and the
lone-.safetensors parent-dir scan.

Each of these fails on pre-Branch-A master (it is the negative that proves the
change is real):
  * pull's HF classifier matches tags EXACTLY, never by substring (MED-15: a tag
    that merely CONTAINS 'lora'/'vae' no longer misclassifies), and returns
    'unknown' - not a silent 'llm' - when no hard signal resolves;
  * add_local records a deterministically-detected type, and 'unknown' (not 'llm')
    for an HF dir with no hard signal in config.json;
  * a lone .safetensors beside a config.json + tokenizer registers the DIRECTORY as
    an HF model; a lone .safetensors with no siblings is rejected with a PRECISE,
    actionable reason (never the bare "Not a model");
  * a type='unknown' model is runnable when named but is never auto-picked as the
    default chat model; its type is mutable via set_model_type / the CLI.
"""

import json
from pathlib import Path

import pytest

from localm import model_manager as mm
from localm.model_manager.pull import _hf_pipeline_tag_to_type


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
#  pull.py HF classifier: exact tag match (MED-15) + unknown sentinel          #
# --------------------------------------------------------------------------- #

def _patch_hf(monkeypatch, payload):
    import localm.discover as discover
    monkeypatch.setattr(discover, "_get", lambda url, params=None: payload)


def test_hf_tag_substring_does_not_misclassify(monkeypatch):
    # 'exploration' CONTAINS 'lora' as a substring (...p-LORA-tion...). Exact matching
    # must NOT call this repo a LoRA. On master the substring check returns 'lora'.
    _patch_hf(monkeypatch, {"pipeline_tag": "text-generation", "tags": ["exploration"]})
    assert _hf_pipeline_tag_to_type("owner/repo") == "llm"


def test_hf_vae_substring_does_not_misclassify(monkeypatch):
    # A tag merely CONTAINING 'vae' as a substring must not be read as a VAE. On
    # master `"vae" in "vae-experiments".lower()` is True and returns 'vae'.
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


# --------------------------------------------------------------------------- #
#  add_local: lone .safetensors dir-scan + HF-dir detection                    #
# --------------------------------------------------------------------------- #

def test_lone_safetensors_beside_config_registers_directory(tmp_path, isolated_home):
    # A .safetensors is not itself loadable; beside a config.json + tokenizer it is
    # part of an HF model dir -> register the DIRECTORY (master rejects the file).
    d = _hf_dir(tmp_path, "my-llm", architectures=["LlamaForCausalLM"])
    weight = d / "model.safetensors"
    assert mm.add_local(str(weight)) is True
    reg = mm.load_registry()
    assert "my-llm" in reg
    assert Path(reg["my-llm"]["path"]).is_dir()      # the dir, not the bare file
    assert reg["my-llm"]["model_type"] == "llm"      # A1 detection from architectures


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
    # No hard signal in config.json -> 'unknown', never a silent 'llm' (master: 'llm').
    d = _hf_dir(tmp_path, "mystery", architectures=None)
    assert mm.add_local(str(d)) is True
    assert mm.load_registry()["mystery"]["model_type"] == "unknown"


def test_add_local_gguf_still_llm(tmp_path, isolated_home):
    # Guard: a bare .gguf is a llama.cpp text model -> stays 'llm' (no regression).
    f = tmp_path / "plain.gguf"
    f.write_bytes(b"GGUF\x00\x00\x00\x00plain")
    assert mm.add_local(str(f)) is True
    assert mm.load_registry()["plain"]["model_type"] == "llm"


# --------------------------------------------------------------------------- #
#  A2: unknown is runnable-by-name but never auto-picked as the chat default   #
# --------------------------------------------------------------------------- #

def test_is_auto_chat_eligible():
    from localm.model_manager import is_auto_chat_eligible
    assert is_auto_chat_eligible({"model_type": "llm"}) is True
    assert is_auto_chat_eligible({}) is True                       # legacy entry = llm
    assert is_auto_chat_eligible({"model_type": "unknown"}) is False


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


def test_autopick_prefers_llm_over_unknown(monkeypatch):
    from localm.plugins.mcpserver.server import EngineCache
    reg = {"a-unknown": {"path": "x", "model_type": "unknown"},
           "b-llm": {"path": "y", "model_type": "llm"}}
    monkeypatch.setattr("localm.config.load_registry", lambda: reg)
    assert EngineCache(default_model=None).resolve_model(None) == "b-llm"


# --------------------------------------------------------------------------- #
#  A2: type is mutable (set_model_type helper + CLI)                           #
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
