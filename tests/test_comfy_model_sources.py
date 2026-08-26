# SPDX-License-Identifier: AGPL-3.0-or-later
"""COMFY_MODEL_SOURCES: the curated filename -> HuggingFace source table that
offers a download when ComfyUI preflight detects a missing workflow model.
Exact-filename lookup only (no fuzzy matching) - a different keyspace from
MODEL_SHORTCUTS (a user-typed alias for `localm pull <alias>`, not an installed
filename ComfyUI reports)."""

from localm.model_manager.registry import (
    COMFY_MODEL_SOURCES,
    ComfySource,
    resolve_comfy_model_source,
)


class TestResolveComfyModelSource:
    def test_hits_every_curated_filename(self):
        for filename in COMFY_MODEL_SOURCES:
            source = resolve_comfy_model_source(filename)
            assert isinstance(source, ComfySource)
            assert source.spec.endswith(f":{filename}")
            assert source.size_bytes > 0
            assert source.model_type
            assert source.comfy_subfolder

    def test_miss_for_an_arbitrary_filename(self):
        assert resolve_comfy_model_source("not-a-known-model.safetensors") is None
        assert resolve_comfy_model_source("") is None

    def test_is_exact_match_only_not_fuzzy(self):
        # A near-miss (different case, or a substring) does not resolve: lookup
        # is curated exact-filename only, never a heuristic guess.
        assert resolve_comfy_model_source("Clip_L.safetensors") is None
        assert resolve_comfy_model_source("clip_l") is None

    def test_flux_unet_source(self):
        source = resolve_comfy_model_source("flux1-dev-Q8_0.gguf")
        assert source.spec == "city96/FLUX.1-dev-gguf:flux1-dev-Q8_0.gguf"
        assert source.model_type == "diffusion-unet"
        assert source.comfy_subfolder == "unet"

    def test_flux_text_encoders_share_a_repo(self):
        clip_l = resolve_comfy_model_source("clip_l.safetensors")
        t5xxl = resolve_comfy_model_source("t5xxl_fp8_e4m3fn.safetensors")
        assert clip_l.spec.startswith("comfyanonymous/flux_text_encoders:")
        assert t5xxl.spec.startswith("comfyanonymous/flux_text_encoders:")
        assert clip_l.model_type == t5xxl.model_type == "text-encoder"
        assert clip_l.comfy_subfolder == t5xxl.comfy_subfolder == "clip"

    def test_flux_vae_sourced_from_ungated_repo(self):
        # ae.safetensors is byte-identical in the gated FLUX.1-dev repo and the
        # ungated FLUX.1-schnell repo, and is sourced from the ungated one.
        source = resolve_comfy_model_source("ae.safetensors")
        assert source.spec == "black-forest-labs/FLUX.1-schnell:ae.safetensors"
        assert "FLUX.1-dev" not in source.spec
        assert source.model_type == "vae"
        assert source.comfy_subfolder == "vae"
