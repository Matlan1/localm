"""docs/models.md is generated from localm's own model tables and must match them."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gen():
    spec = importlib.util.spec_from_file_location("gen_models_page", ROOT / "scripts" / "gen_models_page.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_models_page_matches_the_tables():
    gen = _gen()
    assert (ROOT / "docs" / "models.md").read_text(encoding="utf-8") == gen.render(), (
        "docs/models.md is out of date: run python scripts/gen_models_page.py")


def test_models_page_names_a_table_entry():
    from localm.model_manager import gguf
    text = _gen().render()
    assert all(f"`{a}`" in text for a in gguf.GGUF_DIFFUSION_ARCHITECTURES)
