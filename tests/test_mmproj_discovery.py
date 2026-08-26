# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GGUF model keeps its vision projector (mmproj) across a GUI or registry model
switch. These cover the discovery the server uses: an explicit registry 'mmproj',
else a sibling projector auto-detected next to the GGUF. The end-to-end mtmd
vision path is covered separately.
"""

import localm.model_manager as mm
from localm.model_manager import find_sibling_mmproj, get_model_mmproj


def _gguf(p):
    p.write_bytes(b"GGUF\x00")
    return p


class TestFindSiblingMmproj:
    def test_single_sibling_detected(self, tmp_path):
        model = _gguf(tmp_path / "gemma-3-4b-it-Q8_0.gguf")
        proj = _gguf(tmp_path / "mmproj-gemma-3-4b-it-f16.gguf")
        assert find_sibling_mmproj(model) == proj

    def test_no_sibling_returns_none(self, tmp_path):
        model = _gguf(tmp_path / "qwen-7b-Q4.gguf")
        _gguf(tmp_path / "another-model-Q4.gguf")   # not an mmproj
        assert find_sibling_mmproj(model) is None

    def test_non_gguf_model_returns_none(self, tmp_path):
        d = tmp_path / "hf-model"
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        _gguf(tmp_path / "mmproj-x.gguf")
        assert find_sibling_mmproj(d) is None

    def test_excludes_the_model_file_itself(self, tmp_path):
        # A model whose own name contains 'mmproj' must not match itself.
        model = _gguf(tmp_path / "weird-mmproj-model-Q8.gguf")
        assert find_sibling_mmproj(model) is None

    def test_ambiguous_without_stem_match_returns_none(self, tmp_path):
        model = _gguf(tmp_path / "gemma-3-4b-Q8.gguf")
        _gguf(tmp_path / "mmproj-qwen-f16.gguf")
        _gguf(tmp_path / "mmproj-llava-f16.gguf")
        assert find_sibling_mmproj(model) is None   # do not guess

    def test_ambiguous_resolved_by_stem(self, tmp_path):
        model = _gguf(tmp_path / "gemma-3-4b-Q8.gguf")
        proj = _gguf(tmp_path / "mmproj-gemma-3-4b-f16.gguf")
        _gguf(tmp_path / "mmproj-qwen-f16.gguf")
        assert find_sibling_mmproj(model) == proj


class TestGetModelMmproj:
    def test_registry_explicit_mmproj_wins(self, tmp_path, monkeypatch):
        model = _gguf(tmp_path / "m.gguf")
        proj = _gguf(tmp_path / "my-explicit-proj.gguf")
        monkeypatch.setattr(mm, "load_registry", lambda: {
            "m": {"path": str(model), "source": "local", "mmproj": str(proj)}})
        assert get_model_mmproj("m") == str(proj)

    def test_registry_mmproj_missing_falls_back_to_sibling(self, tmp_path, monkeypatch):
        model = _gguf(tmp_path / "gemma-Q8.gguf")
        sibling = _gguf(tmp_path / "mmproj-gemma-f16.gguf")
        monkeypatch.setattr(mm, "load_registry", lambda: {
            "g": {"path": str(model), "source": "local",
                  "mmproj": str(tmp_path / "gone.gguf")}})  # recorded but deleted
        assert get_model_mmproj("g") == str(sibling)

    def test_sibling_autodetect_for_registered_gguf(self, tmp_path, monkeypatch):
        model = _gguf(tmp_path / "gemma-Q8.gguf")
        sibling = _gguf(tmp_path / "mmproj-gemma-f16.gguf")
        monkeypatch.setattr(mm, "load_registry", lambda: {
            "g": {"path": str(model), "source": "local"}})
        assert get_model_mmproj("g") == str(sibling)

    def test_none_when_no_projector(self, tmp_path, monkeypatch):
        model = _gguf(tmp_path / "text-only-Q8.gguf")
        monkeypatch.setattr(mm, "load_registry", lambda: {
            "t": {"path": str(model), "source": "local"}})
        assert get_model_mmproj("t") is None

    def test_unknown_model_returns_none(self, monkeypatch):
        monkeypatch.setattr(mm, "load_registry", lambda: {})
        assert get_model_mmproj("does-not-exist") is None
