# SPDX-License-Identifier: AGPL-3.0-or-later
"""A GGUF model keeps its vision projector (mmproj) across a GUI or registry model
switch. These cover the discovery the server uses: an explicit registry 'mmproj',
else a sibling projector auto-detected next to the GGUF. The end-to-end mtmd
vision path is covered separately.
"""

import struct

import pytest

import localm.model_manager as mm
from localm.model_manager import find_sibling_mmproj, get_model_mmproj


def _gguf(p):
    p.write_bytes(b"GGUF\x00")
    return p


_T_UINT32 = 4
_T_STRING = 8


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path, kv):
    """Write a GGUF v3 header carrying the metadata *kv* and no tensors."""
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(kv))]
    for key, vtype, val in kv:
        out.append(_s(key))
        out.append(struct.pack("<I", vtype))
        out.append(_s(val) if vtype == _T_STRING else struct.pack("<I", val))
    path.write_bytes(b"".join(out))
    return path


def _real_text_model_gguf(path, architecture: str, embedding_length: int):
    """A minimal but REAL GGUF header for a text model: general.architecture
    plus its embedding_length, the exact two keys gguf_n_embd reads. Ground-
    truthed against a real Qwen2.5-Coder-7B-Instruct GGUF, which reports
    general.architecture='qwen2' and qwen2.embedding_length=3584."""
    return _write_gguf(path, [
        ("general.architecture", _T_STRING, architecture),
        (f"{architecture}.embedding_length", _T_UINT32, embedding_length)])


def _real_mmproj_gguf(path, projection_dim: int):
    """A minimal but REAL GGUF header for a clip mmproj: general.architecture
    plus clip.vision.projection_dim, the exact two keys gguf_n_embd reads for
    a clip file. Ground-truthed against a real mmproj-*-F16.gguf, which
    reports general.architecture='clip' and clip.vision.projection_dim=5120.
    openbmb's MiniCPM-V-2_6 mmproj-model-f16.gguf reports projection_dim=0,
    which gguf_n_embd reads as unknown."""
    return _write_gguf(path, [
        ("general.architecture", _T_STRING, "clip"),
        ("clip.vision.projection_dim", _T_UINT32, projection_dim)])


def _real_imatrix_gguf(path):
    """A minimal but REAL GGUF header for a llama.cpp importance-matrix file.
    Ground-truthed against a real imatrix.gguf, which reports
    general.type='imatrix' and carries no general.architecture."""
    return _write_gguf(path, [
        ("general.type", _T_STRING, "imatrix"),
        ("imatrix.chunk_count", _T_UINT32, 934)])


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

    def test_lone_sibling_with_mismatched_embedding_width_is_not_attached(self, tmp_path):
        """The live regression: a leftover mmproj for a DIFFERENT, larger
        model (here: projection_dim=5120, matching the real
        mmproj-Qwen3.8-27B-Uncensored-F16.gguf from the bug report) sits
        alone next to an unrelated 7B model (embedding_length=3584, matching
        the real Qwen2.5-Coder-7B-Instruct-Q6_K.gguf from the same report).
        Being the ONLY mmproj-looking file in the directory must not be
        enough to auto-attach it - that is exactly how llama.cpp's own
        mtmd_init_from_file ends up failing with "mismatch between text
        model (n_embd = 3584) and mmproj (n_embd = 5120)" after the load was
        already attempted."""
        model = _real_text_model_gguf(
            tmp_path / "Qwen2.5-Coder-7B-Instruct-Q6_K.gguf", "qwen2", 3584)
        _real_mmproj_gguf(
            tmp_path / "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", 5120)
        assert find_sibling_mmproj(model) is None

    def test_lone_sibling_with_matching_embedding_width_is_attached(self, tmp_path):
        """The legitimate case the fix must not break: a lone mmproj whose
        projection_dim genuinely matches its model's embedding_length is
        still auto-attached."""
        model = _real_text_model_gguf(
            tmp_path / "some-vl-model-Q8.gguf", "qwen2", 5120)
        proj = _real_mmproj_gguf(tmp_path / "mmproj-some-vl-model-f16.gguf", 5120)
        assert find_sibling_mmproj(model) == proj

    def test_lone_sibling_with_unreadable_header_is_still_attached(self, tmp_path):
        """gguf_n_embd() returns None (unknown) for the placeholder
        b"GGUF\\x00" fixtures every other test in this class uses - that
        must fall through to the pre-existing behavior, not be treated as a
        mismatch. Guards against the fix regressing every other test here
        that never bothered writing a real header."""
        model = _gguf(tmp_path / "gemma-3-4b-it-Q8_0.gguf")
        proj = _gguf(tmp_path / "mmproj-gemma-3-4b-it-f16.gguf")
        assert find_sibling_mmproj(model) == proj

    def test_matching_stem_with_mismatched_embedding_width_is_not_attached(self, tmp_path):
        """A projector whose name carries the model's own name is still refused
        when its projection_dim differs from the model's embedding_length."""
        model = _real_text_model_gguf(
            tmp_path / "some-vl-model-Q8.gguf", "qwen2", 3584)
        _real_mmproj_gguf(tmp_path / "mmproj-some-vl-model-f16.gguf", 5120)
        assert find_sibling_mmproj(model) is None

    def test_projector_named_for_another_model_in_the_folder_is_not_attached(self, tmp_path):
        """A lone projector named after a different model that sits in the same
        folder belongs to that model, not to its neighbour."""
        llama = _gguf(tmp_path / "Llama3.3-8B-Instruct-Thinking.gguf")
        qwen = _gguf(tmp_path / "Qwen3.8-27B-Uncensored-Q4_K_M.gguf")
        proj = _gguf(tmp_path / "mmproj-Qwen3.8-27B-Uncensored-F16.gguf")
        assert find_sibling_mmproj(llama) is None
        assert find_sibling_mmproj(qwen) == proj

    def test_projector_named_for_an_absent_model_is_attached_when_widths_match(
            self, tmp_path):
        """A projector named after a model that is not in the folder, such as a
        fine-tune stored next to its base model's projector, pairs like a
        generically named one."""
        model = _real_text_model_gguf(
            tmp_path / "Cydonia-24B-v4.1-Q4_K_M.gguf", "llama", 5120)
        proj = _real_mmproj_gguf(
            tmp_path / "mmproj-Mistral-Small-3.2-24B-Instruct-2506-f16.gguf", 5120)
        assert find_sibling_mmproj(model) == proj

    def test_projector_carrying_the_models_base_name_is_attached(self, tmp_path):
        """koboldcpp's projector names (``LLaMA3-8B_mmproj-Q4_1.gguf``) do not
        contain the leading token of ``Meta-Llama-3-8B-Instruct``, but the model's
        name contains the projector's."""
        model = _real_text_model_gguf(
            tmp_path / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", "llama", 4096)
        proj = _real_mmproj_gguf(tmp_path / "LLaMA3-8B_mmproj-Q4_1.gguf", 4096)
        assert find_sibling_mmproj(model) == proj


class TestGenericallyNamedProjector:
    """A projector whose name carries no model name (``mmproj-F16.gguf``,
    ``mmproj-model-f16.gguf``) pairs by the models it sits with."""

    def test_alone_with_its_model_is_attached(self, tmp_path):
        model = _real_text_model_gguf(
            tmp_path / "gemma-3-4b-it-Q4_K_M.gguf", "gemma3", 2560)
        proj = _real_mmproj_gguf(tmp_path / "mmproj-F16.gguf", 2560)
        assert find_sibling_mmproj(model) == proj

    def test_attached_to_minicpm_whose_projector_width_is_unknown(self, tmp_path):
        model = _real_text_model_gguf(
            tmp_path / "MiniCPM-V-2_6-Q4_K_M.gguf", "qwen2", 3584)
        proj = _real_mmproj_gguf(tmp_path / "mmproj-model-f16.gguf", 0)
        assert find_sibling_mmproj(model) == proj

    def test_attached_only_to_the_model_whose_width_matches(self, tmp_path):
        gemma = _real_text_model_gguf(
            tmp_path / "gemma-3-4b-it-Q4_K_M.gguf", "gemma3", 2560)
        qwen = _real_text_model_gguf(
            tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "qwen2", 3584)
        proj = _real_mmproj_gguf(tmp_path / "mmproj-F16.gguf", 2560)
        assert find_sibling_mmproj(gemma) == proj
        assert find_sibling_mmproj(qwen) is None

    def test_mismatched_width_is_not_attached(self, tmp_path):
        model = _real_text_model_gguf(
            tmp_path / "gemma-3-4b-it-Q4_K_M.gguf", "gemma3", 2560)
        _real_mmproj_gguf(tmp_path / "mmproj-F16.gguf", 5120)
        assert find_sibling_mmproj(model) is None

    def test_every_quant_of_the_model_gets_it(self, tmp_path):
        q4 = _real_text_model_gguf(
            tmp_path / "gemma-3-4b-it-Q4_K_M.gguf", "gemma3", 2560)
        q8 = _real_text_model_gguf(
            tmp_path / "gemma-3-4b-it-Q8_0.gguf", "gemma3", 2560)
        proj = _real_mmproj_gguf(tmp_path / "mmproj-model-f16.gguf", 2560)
        assert find_sibling_mmproj(q4) == proj
        assert find_sibling_mmproj(q8) == proj

    def test_mtp_head_and_imatrix_file_do_not_block_it(self, tmp_path):
        """The peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF layout: an MTP head that
        carries the model's architecture and width, and an imatrix GGUF that has
        no architecture, next to the quants and a lone mmproj-BF16.gguf."""
        model = _real_text_model_gguf(
            tmp_path / "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf", "qwen35moe", 2048)
        _real_text_model_gguf(
            tmp_path / "Tiel-Coder-35B-A3B-UD-Q8_K_XL.gguf", "qwen35moe", 2048)
        _real_text_model_gguf(
            tmp_path / "mtp-Tiel-Coder-35B-A3B.gguf", "qwen35moe", 2048)
        _real_imatrix_gguf(tmp_path / "Tiel-Coder-35B-A3B.imatrix.gguf")
        proj = _real_mmproj_gguf(tmp_path / "mmproj-BF16.gguf", 2048)
        assert find_sibling_mmproj(model) == proj

    def test_not_attached_when_a_different_model_in_the_folder_could_use_it(
            self, tmp_path):
        """Same family, same width, different architecture: the projector could
        be either model's, so neither gets it automatically."""
        coder = _real_text_model_gguf(
            tmp_path / "Qwen2.5-Coder-7B-Instruct-Q6_K.gguf", "qwen2", 3584)
        vl = _real_text_model_gguf(
            tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "qwen2vl", 3584)
        _real_mmproj_gguf(tmp_path / "mmproj-F16.gguf", 3584)
        assert find_sibling_mmproj(coder) is None
        assert find_sibling_mmproj(vl) is None

    def test_a_model_of_another_width_reads_no_other_models_header(
            self, tmp_path, monkeypatch):
        """A model whose width differs from the projector's is refused on those
        two headers; the folder's other models are not read."""
        from localm.model_manager import registry
        qwen = _real_text_model_gguf(
            tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "qwen2", 3584)
        _real_text_model_gguf(tmp_path / "gemma-3-4b-it-Q4_K_M.gguf", "gemma3", 2560)
        _real_text_model_gguf(tmp_path / "Phi-2-Q4_K_M.gguf", "phi2", 2560)
        _real_mmproj_gguf(tmp_path / "mmproj-F16.gguf", 2560)
        read = []
        real_fit = registry._gguf_fit

        def counting_fit(path):
            read.append(path.name)
            return real_fit(path)

        monkeypatch.setattr(registry, "_gguf_fit", counting_fit)

        assert find_sibling_mmproj(qwen) is None
        assert sorted(read) == ["Qwen2.5-7B-Instruct-Q4_K_M.gguf", "mmproj-F16.gguf"]

    def test_unknown_projector_width_with_another_model_is_not_attached(self, tmp_path):
        """A projector whose width cannot be read cannot rule any other model
        in the folder out."""
        minicpm = _real_text_model_gguf(
            tmp_path / "MiniCPM-V-2_6-Q4_K_M.gguf", "qwen2", 3584)
        coder = _real_text_model_gguf(
            tmp_path / "Qwen2.5-Coder-7B-Instruct-Q6_K.gguf", "qwen2", 3584)
        _real_mmproj_gguf(tmp_path / "mmproj-model-f16.gguf", 0)
        assert find_sibling_mmproj(minicpm) is None
        assert find_sibling_mmproj(coder) is None


class TestPickMmprojCandidate:
    def test_empty_candidate_list_returns_none(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        assert _pick_mmproj_candidate("gemma-3-4b", []) is None

    def test_lone_candidate_matching_stem_returns_name(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        assert _pick_mmproj_candidate(
            "gemma-3-4b-it", ["mmproj-gemma-3-4b-it-f16.gguf"]) == "mmproj-gemma-3-4b-it-f16.gguf"

    def test_lone_candidate_named_for_another_listed_model_returns_none(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        assert _pick_mmproj_candidate(
            "Llama3.3-8B-Instruct", ["mmproj-Qwen3.8-27B-Uncensored-F16.gguf"],
            others=["Qwen3.8-27B-Uncensored-Q4_K_M.gguf"]) is None

    def test_lone_generic_candidate_is_returned(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        assert _pick_mmproj_candidate(
            "gemma-3-4b-it", ["mmproj-F16.gguf"]) == "mmproj-F16.gguf"

    def test_lone_candidate_carrying_the_models_base_name_is_returned(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        assert _pick_mmproj_candidate(
            "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            ["LLaMA3-8B_mmproj-Q4_1.gguf"]) == "LLaMA3-8B_mmproj-Q4_1.gguf"

    def test_lone_generic_candidate_a_listed_model_could_use_returns_none(self):
        from localm.model_manager.registry import _GgufFit, _pick_mmproj_candidate
        fits = {"gemma-3-4b-it-Q4_K_M.gguf": _GgufFit("gemma3", 2560),
                "Phi-2-Q4_K_M.gguf": _GgufFit("phi2", 2560),
                "mmproj-F16.gguf": _GgufFit("clip", 2560)}
        assert _pick_mmproj_candidate(
            "Phi-2-Q4_K_M.gguf", ["mmproj-F16.gguf"],
            others=["gemma-3-4b-it-Q4_K_M.gguf"], fit=fits.__getitem__) is None

    def test_a_projector_never_gets_a_projector(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        assert _pick_mmproj_candidate("mmproj-model-f16", ["mmproj-F16.gguf"]) is None

    def test_multiple_candidates_single_stem_match(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        cands = ["mmproj-qwen-f16.gguf", "mmproj-gemma-f16.gguf", "mmproj-llava-f16.gguf"]
        assert _pick_mmproj_candidate("gemma-3-4b", cands) == "mmproj-gemma-f16.gguf"

    def test_multiple_candidates_no_stem_match(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        cands = ["mmproj-qwen-f16.gguf", "mmproj-llava-f16.gguf"]
        assert _pick_mmproj_candidate("gemma-3-4b", cands) is None

    def test_multiple_candidates_ambiguous_stem_matches_returns_none(self):
        from localm.model_manager.registry import _pick_mmproj_candidate
        cands = ["mmproj-gemma-f16.gguf", "mmproj-gemma-q4.gguf"]
        assert _pick_mmproj_candidate("gemma-3-4b", cands) is None


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


_PAIRING_CASES = [
    (["gemma-3-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf"],
     "gemma-3-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf"),
    (["MiniCPM-V-2_6-Q4_K_M.gguf", "mmproj-model-f16.gguf"],
     "MiniCPM-V-2_6-Q4_K_M.gguf", "mmproj-model-f16.gguf"),
    (["gemma-3-4b-it-Q8_0.gguf", "mmproj-gemma-3-4b-it-f16.gguf"],
     "gemma-3-4b-it-Q8_0.gguf", "mmproj-gemma-3-4b-it-f16.gguf"),
    (["Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", "LLaMA3-8B_mmproj-Q4_1.gguf"],
     "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", "LLaMA3-8B_mmproj-Q4_1.gguf"),
    (["Llama3.3-8B-Instruct.gguf", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf",
      "mmproj-Qwen3.8-27B-Uncensored-F16.gguf"],
     "Llama3.3-8B-Instruct.gguf", None),
    (["modelA.gguf", "modelB.gguf", "mmproj-modelB-f16.gguf"], "modelA.gguf", None),
]


class TestListingAndFolderAgree:
    """`localm pull` picks from a repo's file listing and the folder scan picks
    from a directory, through the same policy. With no readable GGUF header on
    either side, the same file names give the same answer."""

    @pytest.mark.parametrize("files,model,expected", _PAIRING_CASES)
    def test_same_names_same_pick(self, tmp_path, files, model, expected):
        from localm.model_manager.pull import _pick_mmproj_from_listing
        folder = tmp_path / "folder"
        folder.mkdir()
        for name in files:
            _gguf(folder / name)
        base = tmp_path / "models"
        base.mkdir()

        from_folder = find_sibling_mmproj(folder / model)
        from_listing = _pick_mmproj_from_listing(files, model, base)

        assert (from_folder.name if from_folder else None) == expected
        assert from_listing == expected


# GGUF file names of real HuggingFace repos, from the HF API on 2026-09-25.
_REAL_LISTINGS = [
    (["gemma-3-4b-it-Q3_K_L.gguf", "gemma-3-4b-it-Q4_K_M.gguf", "gemma-3-4b-it-Q6_K.gguf",
      "gemma-3-4b-it-Q8_0.gguf", "mmproj-model-f16.gguf"],
     "gemma-3-4b-it-Q4_K_M.gguf", "mmproj-model-f16.gguf"),
    (["gemma-3-4b-it-Q4_K_M.gguf", "gemma-3-4b-it-Q8_0.gguf", "gemma-3-4b-it-f16.gguf",
      "mmproj-model-f16.gguf"],
     "gemma-3-4b-it-Q8_0.gguf", "mmproj-model-f16.gguf"),
    (["ggml-model-IQ3_M.gguf", "ggml-model-IQ4_XS.gguf", "ggml-model-Q4_K_M.gguf",
      "ggml-model-Q8_0.gguf", "ggml-model-f16.gguf", "mmproj-model-f16.gguf"],
     "ggml-model-Q4_K_M.gguf", "mmproj-model-f16.gguf"),
    (["llava-1.6-mistral-7b.Q6_K.gguf", "llava-1.6-mistral-7b.Q8_0.gguf",
      "llava-v1.6-mistral-7b.Q3_K.gguf", "llava-v1.6-mistral-7b.Q4_K_M.gguf",
      "llava-v1.6-mistral-7b.Q8_0.gguf", "mmproj-model-f16.gguf"],
     "llava-v1.6-mistral-7b.Q4_K_M.gguf", "mmproj-model-f16.gguf"),
    (["Tiel-Coder-35B-A3B-UD-IQ3_XXS.gguf", "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
      "Tiel-Coder-35B-A3B-UD-Q8_K_XL.gguf", "Tiel-Coder-35B-A3B.imatrix.gguf",
      "mmproj-BF16.gguf", "mtp-Tiel-Coder-35B-A3B.gguf"],
     "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"),
    (["imatrix.gguf", "main.bf16.gguf", "mmproj.f16.gguf", "q4_k_m.gguf"],
     "q4_k_m.gguf", "mmproj.f16.gguf"),
    (["Tess-4-27B-Q4_K_M.gguf", "Tess-4-27B-Q6_K.gguf", "Tess-4-27B-Q8_0.gguf",
      "mmproj-Tess-4-27B-F16.gguf", "mtp-Tess-4-27B-Q4_K_M.gguf",
      "mtp-Tess-4-27B-Q8_0.gguf"],
     "Tess-4-27B-Q4_K_M.gguf", "mmproj-Tess-4-27B-F16.gguf"),
]


class TestRealRepoListings:
    """The projector `localm pull` picks from real repo layouts: generic names
    (lmstudio-community, ggml-org, openbmb, cjpais), an MTP head and an imatrix
    file beside the quants (peculiar-ragdoll), generic model names (Hcompany),
    and a named projector beside MTP heads (migtissera)."""

    @pytest.mark.parametrize("files,model,expected", _REAL_LISTINGS)
    def test_pick(self, tmp_path, files, model, expected):
        from localm.model_manager.pull import _pick_mmproj_from_listing
        assert _pick_mmproj_from_listing(files, model, tmp_path) == expected
