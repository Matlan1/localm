"""docs/models.md lists every model kind localm runs, generated from localm's own tables."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gen():
    spec = importlib.util.spec_from_file_location("gen_models_page", ROOT / "scripts" / "gen_models_page.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _page():
    return (ROOT / "docs" / "models.md").read_text(encoding="utf-8")


def test_page_matches_the_tables_and_the_pinned_llama_cpp():
    gen = _gen()
    rec = gen.recorded_arches(_page())
    assert rec is not None, "docs/models.md has no recorded llama.cpp architecture list"
    from localm.setup_llama import pins
    assert rec[0] == pins._PINNED_TAG, "the llama.cpp pin moved: run python scripts/gen_models_page.py"
    assert _page() == gen.render(rec[1], rec[0]), "docs/models.md is out of date: run python scripts/gen_models_page.py"


def test_every_model_type_is_described():
    from localm.model_manager import registry
    gen = _gen()
    assert registry.MODEL_TYPES - {"unknown"} <= set(gen.TYPE_TEXT)


def test_every_plugin_backend_and_media_runtime_is_covered():
    covered = set(_gen().COVERED)
    found = {f"plugins/builtin/{p.name}" for p in (ROOT / "localm/plugins/builtin").iterdir()
             if p.is_dir() and not p.name.startswith("__")}
    found |= {f"media/{p.name}" for p in (ROOT / "localm/media").iterdir()
              if p.is_dir() and not p.name.startswith("__")}
    for f in (ROOT / "localm/inference/backends").glob("*.py"):
        if re.search(r"^class \w*Backend\(", f.read_text(encoding="utf-8"), re.M):
            found.add(f"inference/backends/{f.name}")
    missing = sorted(found - covered)
    assert not missing, f"not on docs/models.md (add to COVERED in scripts/gen_models_page.py): {missing}"


def test_registry_types_every_architecture_the_hf_worker_generates_with():
    from localm.inference.backends import _hf_worker
    from localm.model_manager import registry
    assert set(_hf_worker._GENERATIVE_ARCH_SUFFIXES) == set(registry._HF_LLM_ARCH_SUFFIXES)


def test_an_image_text_to_text_folder_registers_as_a_language_model(tmp_path):
    import json
    from localm.model_manager import registry
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Gemma3ForImageTextToText"]}), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert registry._detect_local_model_type(tmp_path, is_gguf=False, is_hf=True)[0] == "llm"
