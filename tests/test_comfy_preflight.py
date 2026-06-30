# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pre-submit model validation against ComfyUI /object_info (I3 / MEDIA-1).

The preflight confirms each loader's model file exists BEFORE the chat model is
unloaded: it names a genuinely-missing file, auto-substitutes a single unambiguous
precision variant, leaves enum (non-model) combos alone, and is a no-op when
/object_info cannot be read (so it never breaks a working setup).
"""

from unittest.mock import MagicMock, patch

from localm.image_gen import comfy
# preflight_models / comfy_object_info now live in the shared client; the
# generic ComfyUI plumbing moved out of image_gen.comfy into here. preflight
# calls comfy_object_info as a bare global in this module, so the stub that
# replaces /object_info must be patched on comfy_client (the symbol's home),
# not on the image_gen.comfy re-export. (Re-exports stay importable; this only
# follows the moved symbol for the INTERNAL call.)
from localm.media import comfy_client


def _object_info(unet_options):
    """A minimal /object_info map for a UNETLoader with the given file options."""
    return {
        "UNETLoader": {"input": {"required": {
            "unet_name": [list(unet_options), {"tooltip": "the unet"}],
            "weight_dtype": [["default", "fp8_e4m3fn", "fp8_e5m2"]],
        }}},
        "KSampler": {"input": {"required": {
            "sampler_name": [["euler", "uni_pc", "dpmpp_2m"]],
            "scheduler": [["simple", "normal"]],
            "seed": ["INT"], "steps": ["INT"], "cfg": ["FLOAT"],
        }}},
    }


def _wan_unet_workflow(unet_name="wan2.2_ti2v_5B_fp16.safetensors"):
    return {
        "1": {"inputs": {"unet_name": unet_name, "weight_dtype": "default"},
              "class_type": "UNETLoader"},
        "8": {"inputs": {"seed": 0, "steps": 30, "cfg": 5.0,
                         "sampler_name": "uni_pc", "scheduler": "simple"},
              "class_type": "KSampler"},
    }


class TestComboHelpers:
    def test_combo_options_extracts_choices(self):
        spec = {"input": {"required": {"unet_name": [["a.safetensors", "b.safetensors"], {}]}}}
        assert comfy._combo_options(spec, "unet_name") == ["a.safetensors", "b.safetensors"]

    def test_combo_options_none_for_typed_input(self):
        spec = {"input": {"required": {"seed": ["INT"]}}}
        assert comfy._combo_options(spec, "seed") is None

    def test_looks_like_model_files(self):
        assert comfy._looks_like_model_files(["x.safetensors", "y.gguf"]) is True
        assert comfy._looks_like_model_files(["euler", "dpmpp_2m"]) is False
        assert comfy._looks_like_model_files([]) is False

    def test_normalize_strips_precision(self):
        a = comfy._normalize_model_base("wan2.2_ti2v_5B_fp16.safetensors")
        b = comfy._normalize_model_base("wan2.2_ti2v_5B_fp8_scaled.safetensors")
        assert a == b and a                      # same base, precision-insensitive

    def test_normalize_keeps_distinct_models_distinct(self):
        a = comfy._normalize_model_base("sd3.5_medium.safetensors")
        b = comfy._normalize_model_base("sd3.5_large.safetensors")
        assert a != b

    def test_normalize_keeps_version_discriminators(self):
        # Bare digits / lone letters are version/variant markers, NOT precision -
        # they must keep genuinely different models apart (regression: these used to
        # collapse and trigger a wrong cross-version substitution).
        assert (comfy._normalize_model_base("wan2.1_ti2v_5B_fp16.safetensors")
                != comfy._normalize_model_base("wan2.2_ti2v_5B_fp16.safetensors"))
        assert (comfy._normalize_model_base("model_s.safetensors")
                != comfy._normalize_model_base("model_m.safetensors"))
        assert (comfy._normalize_model_base("vae_1.safetensors")
                != comfy._normalize_model_base("vae_2.safetensors"))

    def test_no_cross_version_substitution(self):
        # Workflow wants Wan 2.2; only Wan 2.1 is installed -> NOT a precision variant,
        # so it must be reported missing, never silently swapped to the other version.
        wf = _wan_unet_workflow("wan2.2_ti2v_5B_fp16.safetensors")
        info = _object_info(["wan2.1_ti2v_5B_fp16.safetensors"])
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = comfy.preflight_models(wf, "http://x")
        assert not ok
        assert wf["1"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"


class TestPreflight:
    def test_present_model_passes_unchanged(self):
        wf = _wan_unet_workflow()
        info = _object_info(["wan2.2_ti2v_5B_fp16.safetensors", "other.safetensors"])
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = comfy.preflight_models(wf, "http://x")
        assert ok and msg == ""
        assert wf["1"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"

    def test_missing_with_one_variant_substitutes(self):
        wf = _wan_unet_workflow()
        progress = []
        info = _object_info(["wan2.2_ti2v_5B_fp8_scaled.safetensors"])
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = comfy.preflight_models(wf, "http://x",
                                             on_progress=progress.append)
        assert ok and msg == ""
        assert wf["1"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp8_scaled.safetensors"
        assert any("substituting" in p for p in progress)

    def test_missing_with_no_variant_fails_naming_file(self):
        wf = _wan_unet_workflow()
        info = _object_info(["totally_different_model.safetensors"])
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = comfy.preflight_models(wf, "http://x")
        assert not ok
        assert "wan2.2_ti2v_5B_fp16.safetensors" in msg
        assert "UNETLoader" in msg
        assert "Workflow panel" in msg
        # not substituted to an unrelated model
        assert wf["1"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"

    def test_ambiguous_variants_not_substituted(self):
        wf = _wan_unet_workflow()
        # two precision variants -> ambiguous -> report missing, never guess
        info = _object_info(["wan2.2_ti2v_5B_fp8_scaled.safetensors",
                             "wan2.2_ti2v_5B_bf16.safetensors"])
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = comfy.preflight_models(wf, "http://x")
        assert not ok
        assert wf["1"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"

    def test_enum_combo_mismatch_not_flagged(self):
        # KSampler.sampler_name is an enum, not a model file: a value not in the
        # list must NOT be reported as a missing model.
        wf = _wan_unet_workflow()
        wf["8"]["inputs"]["sampler_name"] = "uni_pc"        # valid file present too
        info = _object_info(["wan2.2_ti2v_5B_fp16.safetensors"])
        with patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = comfy.preflight_models(wf, "http://x")
        assert ok and msg == ""

    def test_unreachable_object_info_is_noop(self):
        wf = _wan_unet_workflow("does-not-exist.safetensors")
        with patch.object(comfy_client, "comfy_object_info", return_value=None):
            ok, msg = comfy.preflight_models(wf, "http://x")
        assert ok and msg == ""                              # best-effort, never blocks

    def test_object_info_fetch_handles_network_error(self):
        with patch.object(comfy.urllib.request, "urlopen",
                          side_effect=OSError("refused")):
            assert comfy.comfy_object_info("http://127.0.0.1:9") is None


class TestPreflightBeforeUnload:
    """The whole point: a missing model fails BEFORE the chat model unload."""

    def test_video_missing_model_fails_before_unload(self, tmp_path):
        from localm.video_gen import comfy as vcomfy
        unload_spy = MagicMock()
        info = _object_info(["unrelated_other.safetensors"])
        with patch.object(vcomfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(vcomfy, "_localm_unload", unload_spy), \
             patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = vcomfy.generate_video("a fox", tmp_path / "out.mp4")
        assert not ok
        assert "wan2.2_ti2v_5B_fp16.safetensors" in msg     # the exact missing file
        assert "Workflow panel" in msg
        unload_spy.assert_not_called()                       # no pointless unload

    def test_music_missing_checkpoint_fails_before_unload(self, tmp_path):
        from localm.music_gen import comfy as mcomfy
        unload_spy = MagicMock()
        info = {"CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": [["some_other_checkpoint.safetensors"]]}}}}
        with patch.object(mcomfy, "ensure_comfy", return_value=(True, "up")), \
             patch.object(mcomfy, "_localm_unload", unload_spy), \
             patch.object(comfy_client, "comfy_object_info", return_value=info):
            ok, msg = mcomfy.generate_music("synthwave", tmp_path / "out.flac")
        assert not ok
        assert "ace_step_v1_3.5b.safetensors" in msg
        unload_spy.assert_not_called()
